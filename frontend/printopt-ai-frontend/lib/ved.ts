/**
 * VED = P / (v × h × t)
 * P: Laser Power [W]
 * v: Scan Speed [mm/s]
 * h: Hatch Spacing [mm] (input in µm, converted)
 * t: Layer Thickness [mm] (input in µm, converted)
 * Result: J/mm³
 */
export function calculateVED(
  laserPower_W: number,
  scanSpeed_mm_s: number,
  hatchSpacing_um: number,
  layerThickness_um: number
): number {
  const h_mm = hatchSpacing_um / 1000;
  const t_mm = layerThickness_um / 1000;
  if (scanSpeed_mm_s <= 0 || h_mm <= 0 || t_mm <= 0) return 0;
  return laserPower_W / (scanSpeed_mm_s * h_mm * t_mm);
}

export function formatVED(ved: number): string {
  return ved.toFixed(1);
}
