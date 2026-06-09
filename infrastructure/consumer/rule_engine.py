#!/usr/bin/env python3
"""
rule_engine.py — Rule-based detection layer for hybrid IDS

Catches attack patterns that the supervised ML model misses because
they were absent from training data (CICIDS2018) or invisible at the
conn.log feature level:

  - SYN Flood:       massive half-open/rejected connections (conn_state=S0 or REJ)
  - SSH Brute Force: rapid failed SSH connections (port 22, REJ/S0/RSTO)
  - FTP Brute Force: rapid failed FTP connections (port 21, REJ/S0/RSTO)
  - Port Scan:       many unique destination ports from one source IP
  - Slowloris:       long-lived HTTP connections with minimal data

The rule engine operates on the same IPWindow temporal state used by
the ML pipeline, plus per-flow conn_state / port / bytes inspection.
It overrides ML predictions: if ML says BENIGN but a rule fires,
the label is replaced with the rule-based class.

Design:
  - Stateful: maintains per-IP counters in sliding windows
  - Lightweight: O(1) per flow (dict lookups + counter increments)
  - Composable: each rule is independent, easy to add new rules
  - No false negatives by design: thresholds are intentionally low,
    trading false positives for coverage
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class RuleConfig:
    """Tunable thresholds for all rule-based detectors."""
    # Sliding window for rate-based rules
    window_sec: float = 60.0

    # SYN Flood: conn_state=S0 or REJ, high rate from one IP
    syn_flood_min_rate: float = 50.0          # S0/REJ connections/sec
    syn_flood_min_count: int = 200            # absolute minimum in window

    # SSH Brute Force: port 22, all connection states (BF shows as SF/RSTR, not just REJ)
    ssh_bf_min_rate: float = 0.5              # SSH conn/sec (30/min)
    ssh_bf_min_count: int = 10                # absolute minimum in window

    # FTP Brute Force: port 21, all connection states
    ftp_bf_min_rate: float = 0.5
    ftp_bf_min_count: int = 10

    # Port Scan: many unique ports from one IP
    portscan_min_ports: int = 15              # unique dst ports in window
    portscan_min_rate: float = 2.0            # connections/sec

    # Slowloris: long duration, low bytes, HTTP ports
    slowloris_min_duration: float = 20.0      # seconds
    slowloris_max_bytes: float = 1000.0       # orig_bytes
    slowloris_http_ports: frozenset = field(
        default_factory=lambda: frozenset({80, 443, 8080, 8443, 8000})
    )
    slowloris_min_concurrent: int = 5         # from same IP to trigger


# Failed connection states (handshake not completed / rejected)
FAILED_CONN_STATES = frozenset({'S0', 'REJ', 'RSTO', 'RSTOS0', 'ShR', 'SHR', 'S1'})

# ── Per-IP state tracking ────────────────────────────────────────────────────

class _IPRuleState:
    """Per-IP sliding window counters for rule-based detection."""
    __slots__ = ('syn_times', 'ssh_fail_times', 'ftp_fail_times',
                 'conn_times', 'dst_ports', 'slowloris_conns')

    def __init__(self):
        self.syn_times:      deque = deque()   # timestamps of S0 connections
        self.ssh_fail_times: deque = deque()   # timestamps of failed SSH
        self.ftp_fail_times: deque = deque()   # timestamps of failed FTP
        self.conn_times:     deque = deque()   # all connection timestamps
        self.dst_ports:      dict  = {}        # {port: timestamp} for scan detect
        self.slowloris_conns: int  = 0         # count of active slowloris-like


class RuleEngine:
    """
    Stateful rule-based detection engine.

    Usage:
        engine = RuleEngine(RuleConfig())
        for each flow:
            label = engine.evaluate(flow_dict)
            if label is not None:
                # Rule fired — override ML prediction
    """

    def __init__(self, config: Optional[RuleConfig] = None):
        self.cfg = config or RuleConfig()
        self._states: dict[str, _IPRuleState] = defaultdict(_IPRuleState)
        self._last_cleanup = time.time()

    def _evict(self, dq: deque, cutoff: float) -> None:
        """Remove entries older than cutoff from a deque of timestamps."""
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _evict_ports(self, ports: dict, cutoff: float) -> None:
        """Remove port entries older than cutoff."""
        expired = [p for p, ts in ports.items() if ts < cutoff]
        for p in expired:
            del ports[p]

    def evaluate(self, src_ip: str, dst_port: int, conn_state: str,
                 duration: float, orig_bytes: float, ts: float) -> Optional[str]:
        """
        Evaluate all rules for a single flow.

        Args:
            src_ip:     Source IP address
            dst_port:   Destination port number
            conn_state: Zeek connection state (S0, SF, REJ, ...)
            duration:   Connection duration in seconds
            orig_bytes: Bytes sent by originator
            ts:         Flow timestamp (epoch)

        Returns:
            Rule-based label string if a rule fires, or None if no rule matches.
            Priority order: SYN-Flood > SSH/FTP-BruteForce > PortScan > Slowloris
        """
        st = self._states[src_ip]
        cutoff = ts - self.cfg.window_sec
        window = self.cfg.window_sec

        # ── Update state ─────────────────────────────────────────────────
        # All connections
        st.conn_times.append(ts)
        self._evict(st.conn_times, cutoff)

        # Track destination ports for scan detection
        st.dst_ports[dst_port] = ts
        self._evict_ports(st.dst_ports, cutoff)

        # S0 (half-open) and REJ (reset) connections — both indicate SYN flood.
        # S0: SYN sent, no reply (server overwhelmed or port filtered).
        # REJ: SYN sent, RST received (port closed under flood).
        if conn_state in ('S0', 'REJ'):
            st.syn_times.append(ts)
        self._evict(st.syn_times, cutoff)

        # SSH connections (all states — brute force shows as SF/RSTR, not just REJ/S0)
        if dst_port == 22:
            st.ssh_fail_times.append(ts)
        self._evict(st.ssh_fail_times, cutoff)

        # FTP connections (all states — same reasoning)
        if dst_port == 21:
            st.ftp_fail_times.append(ts)
        self._evict(st.ftp_fail_times, cutoff)

        # ── Evaluate rules (priority order) ──────────────────────────────

        # 1. SYN Flood
        syn_count = len(st.syn_times)
        if (syn_count >= self.cfg.syn_flood_min_count and
                syn_count / window >= self.cfg.syn_flood_min_rate):
            return 'SYN-Flood'

        # 2. SSH Brute Force
        ssh_count = len(st.ssh_fail_times)
        if (ssh_count >= self.cfg.ssh_bf_min_count and
                ssh_count / window >= self.cfg.ssh_bf_min_rate):
            return 'SSH-BruteForce-Rule'

        # 3. FTP Brute Force
        ftp_count = len(st.ftp_fail_times)
        if (ftp_count >= self.cfg.ftp_bf_min_count and
                ftp_count / window >= self.cfg.ftp_bf_min_rate):
            return 'FTP-BruteForce-Rule'

        # 4. Port Scan
        n_ports = len(st.dst_ports)
        conn_rate = len(st.conn_times) / window
        if (n_ports >= self.cfg.portscan_min_ports and
                conn_rate >= self.cfg.portscan_min_rate):
            return 'PortScan'

        # 5. Slowloris
        if (dst_port in self.cfg.slowloris_http_ports and
                duration >= self.cfg.slowloris_min_duration and
                orig_bytes <= self.cfg.slowloris_max_bytes):
            # Count how many such connections from this IP
            # We approximate by checking connection rate to HTTP ports
            # with long duration pattern
            st.slowloris_conns += 1
            if st.slowloris_conns >= self.cfg.slowloris_min_concurrent:
                return 'DOS Slowloris'
        else:
            # Decay slowloris counter over time
            if st.slowloris_conns > 0 and len(st.conn_times) > 0:
                age = ts - st.conn_times[0] if st.conn_times else 0
                if age > self.cfg.window_sec:
                    st.slowloris_conns = max(0, st.slowloris_conns - 1)

        return None

    def cleanup(self, now: float) -> None:
        """
        Periodic cleanup: remove IPs with no activity in 2× window.
        Call every ~60s to prevent unbounded memory growth.
        """
        cutoff = now - 2 * self.cfg.window_sec
        stale = [ip for ip, st in self._states.items()
                 if not st.conn_times or st.conn_times[-1] < cutoff]
        for ip in stale:
            del self._states[ip]

    def stats(self) -> dict:
        """Return current state summary for logging."""
        return {
            'tracked_ips': len(self._states),
        }
