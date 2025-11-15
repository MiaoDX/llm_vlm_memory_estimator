"""
Probe results and comparison utilities.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DeviceInfo:
    """GPU device information"""
    device_id: int
    device_name: str
    total_memory_gib: float
    compute_capability: str


@dataclass
class PhaseMetrics:
    """Memory metrics for a single probe phase"""
    phase: str
    max_allocated_gib: float
    max_reserved_gib: float
    elapsed_sec: float

    def __str__(self) -> str:
        return (
            f"{self.phase}: "
            f"allocated={self.max_allocated_gib:.2f} GiB, "
            f"reserved={self.max_reserved_gib:.2f} GiB, "
            f"time={self.elapsed_sec:.2f}s"
        )


@dataclass
class ProbeResult:
    """Results from empirical GPU memory probe"""
    phases: Dict[str, PhaseMetrics]
    device_info: DeviceInfo
    peak_allocated_gib: float
    peak_reserved_gib: float

    def print_summary(self) -> None:
        """Print probe results summary"""
        print("\n" + "=" * 70)
        print("GPU PROBE RESULTS")
        print("=" * 70)
        print(f"\nDevice: {self.device_info.device_name} (ID: {self.device_info.device_id})")
        print(f"Total VRAM: {self.device_info.total_memory_gib:.2f} GiB")
        print(f"Compute Capability: {self.device_info.compute_capability}")

        print("\nPer-Phase Metrics:")
        for phase_name, metrics in self.phases.items():
            print(f"  {metrics}")

        print(f"\nPeak Memory:")
        print(f"  Allocated: {self.peak_allocated_gib:.2f} GiB")
        print(f"  Reserved:  {self.peak_reserved_gib:.2f} GiB")
        print("=" * 70)

    def compare_with_estimate(self, estimated_gib: float) -> "ComparisonReport":
        """
        Compare probe results with estimation.

        Args:
            estimated_gib: Estimated memory from formula

        Returns:
            ComparisonReport with detailed comparison
        """
        diff_gib = self.peak_allocated_gib - estimated_gib
        diff_percent = (diff_gib / estimated_gib) * 100 if estimated_gib > 0 else 0

        # Calculate fragmentation (reserved - allocated)
        fragmentation_gib = self.peak_reserved_gib - self.peak_allocated_gib
        fragmentation_percent = (
            (fragmentation_gib / self.peak_allocated_gib) * 100
            if self.peak_allocated_gib > 0 else 0
        )

        return ComparisonReport(
            estimated_gib=estimated_gib,
            measured_allocated_gib=self.peak_allocated_gib,
            measured_reserved_gib=self.peak_reserved_gib,
            diff_gib=diff_gib,
            diff_percent=diff_percent,
            fragmentation_gib=fragmentation_gib,
            fragmentation_percent=fragmentation_percent,
            device_info=self.device_info,
        )


@dataclass
class ComparisonReport:
    """Comparison between estimated and measured memory"""
    estimated_gib: float
    measured_allocated_gib: float
    measured_reserved_gib: float
    diff_gib: float
    diff_percent: float
    fragmentation_gib: float
    fragmentation_percent: float
    device_info: DeviceInfo

    def print_comparison(self) -> None:
        """Print comparison report"""
        print("\n" + "=" * 70)
        print("ESTIMATION vs PROBE COMPARISON")
        print("=" * 70)
        print(f"\nDevice: {self.device_info.device_name}")
        print(f"\nEstimated Memory:        {self.estimated_gib:>8.2f} GiB")
        print(f"Measured (Allocated):    {self.measured_allocated_gib:>8.2f} GiB")
        print(f"Measured (Reserved):     {self.measured_reserved_gib:>8.2f} GiB")

        print(f"\nDifference:")
        sign = "+" if self.diff_gib >= 0 else ""
        print(f"  Absolute: {sign}{self.diff_gib:>8.2f} GiB")
        print(f"  Relative: {sign}{self.diff_percent:>8.2f}%")

        print(f"\nFragmentation (Reserved - Allocated):")
        print(f"  Absolute: {self.fragmentation_gib:>8.2f} GiB")
        print(f"  Relative: {self.fragmentation_percent:>8.2f}%")

        # Provide interpretation
        print(f"\nInterpretation:")
        if abs(self.diff_percent) < 10:
            print("  ✓ Excellent: Estimation within 10% of actual usage")
        elif abs(self.diff_percent) < 20:
            print("  ✓ Good: Estimation within 20% of actual usage")
        else:
            print("  ⚠ Warning: Estimation differs by >20% from actual usage")
            if self.diff_gib > 0:
                print("    → Actual usage LOWER than estimated (conservative estimate)")
            else:
                print("    → Actual usage HIGHER than estimated (may need calibration)")

        print("=" * 70)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for programmatic access"""
        return {
            "estimated_gib": self.estimated_gib,
            "measured_allocated_gib": self.measured_allocated_gib,
            "measured_reserved_gib": self.measured_reserved_gib,
            "diff_gib": self.diff_gib,
            "diff_percent": self.diff_percent,
            "fragmentation_gib": self.fragmentation_gib,
            "fragmentation_percent": self.fragmentation_percent,
        }
