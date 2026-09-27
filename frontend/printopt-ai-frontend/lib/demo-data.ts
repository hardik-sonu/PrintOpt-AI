import type { DatasetRecord, SourceRecord } from '../types';
import { calculateVED } from './ved';

/** Representative demo records for Ti-6Al-4V LPBF — real literature ranges */
const RAW_RECORDS = [
  { id: 'TI-001', P: 200, v: 1000, t: 30, h: 120, spot: 80, ps: 30, uts: 1050, ys: 980, el: 8.2, src: 'Kok et al. (2018)' },
  { id: 'TI-002', P: 150, v: 800, t: 30, h: 100, spot: 70, ps: 30, uts: 1020, ys: 950, el: 9.1, src: 'Kok et al. (2018)' },
  { id: 'TI-003', P: 250, v: 1200, t: 30, h: 120, spot: 80, ps: 30, uts: 1080, ys: 1010, el: 7.8, src: 'Leuders et al. (2013)' },
  { id: 'TI-004', P: 175, v: 900, t: 40, h: 110, spot: 75, ps: 35, uts: 980, ys: 910, el: 10.5, src: 'Leuders et al. (2013)' },
  { id: 'TI-005', P: 300, v: 1500, t: 30, h: 130, spot: 80, ps: 30, uts: 1100, ys: 1030, el: 7.2, src: 'Xu et al. (2015)' },
  { id: 'TI-006', P: 100, v: 600, t: 25, h: 90, spot: 65, ps: 28, uts: 940, ys: 880, el: 11.0, src: 'Xu et al. (2015)' },
  { id: 'TI-007', P: 200, v: 800, t: 30, h: 100, spot: 80, ps: 30, uts: 1060, ys: 995, el: 8.9, src: 'Murr et al. (2012)' },
  { id: 'TI-008', P: 220, v: 1100, t: 35, h: 120, spot: 80, ps: 32, uts: 1045, ys: 975, el: 8.5, src: 'Murr et al. (2012)' },
  { id: 'TI-009', P: 280, v: 1300, t: 30, h: 125, spot: 85, ps: 30, uts: 1090, ys: 1020, el: 7.6, src: 'Benedetti et al. (2017)' },
  { id: 'TI-010', P: 160, v: 750, t: 30, h: 105, spot: 72, ps: 30, uts: 1010, ys: 940, el: 9.8, src: 'Benedetti et al. (2017)' },
  { id: 'TI-011', P: 350, v: 1600, t: 30, h: 140, spot: 90, ps: 30, uts: 1110, ys: 1040, el: 7.0, src: 'Rafi et al. (2013)' },
  { id: 'TI-012', P: 120, v: 700, t: 30, h: 95, spot: 68, ps: 30, uts: 965, ys: 900, el: 10.8, src: 'Rafi et al. (2013)' },
  { id: 'TI-013', P: 200, v: 1200, t: 30, h: 120, spot: 80, ps: 30, uts: 1030, ys: 960, el: 8.7, src: 'Zhang et al. (2017)' },
  { id: 'TI-014', P: 240, v: 1000, t: 30, h: 110, spot: 80, ps: 30, uts: 1075, ys: 1005, el: 8.0, src: 'Zhang et al. (2017)' },
  { id: 'TI-015', P: 180, v: 850, t: 35, h: 115, spot: 77, ps: 33, uts: 1000, ys: 935, el: 9.5, src: 'Kasperovich et al. (2016)' },
  { id: 'TI-016', P: 260, v: 1250, t: 30, h: 125, spot: 82, ps: 30, uts: 1085, ys: 1015, el: 7.9, src: 'Kasperovich et al. (2016)' },
  { id: 'TI-017', P: 200, v: 1000, t: 25, h: 120, spot: 80, ps: 30, uts: 1055, ys: 985, el: 8.3, src: 'Sallica-Leva et al. (2013)' },
  { id: 'TI-018', P: 200, v: 1000, t: 50, h: 120, spot: 80, ps: 30, uts: 1000, ys: 930, el: 9.0, src: 'Sallica-Leva et al. (2013)' },
  { id: 'TI-019', P: 200, v: 1000, t: 30, h: 80, spot: 80, ps: 30, uts: 1070, ys: 1000, el: 8.1, src: 'Gong et al. (2014)' },
  { id: 'TI-020', P: 200, v: 1000, t: 30, h: 160, spot: 80, ps: 30, uts: 1020, ys: 950, el: 8.8, src: 'Gong et al. (2014)' },
];

export const DEMO_RECORDS: DatasetRecord[] = RAW_RECORDS.map((r) => ({
  record_id: r.id,
  laser_power_w: r.P,
  scan_speed_mm_s: r.v,
  layer_thickness_um: r.t,
  hatch_spacing_um: r.h,
  laser_spot_um: r.spot,
  powder_size_um: r.ps,
  ved_j_mm3: calculateVED(r.P, r.v, r.h, r.t),
  uts_mpa: r.uts,
  yield_strength_mpa: r.ys,
  elongation_pct: r.el,
  source: r.src,
}));

export const DEMO_SOURCES: SourceRecord[] = [
  { id: 'src-001', title: 'Microstructure and mechanical behavior of Ti-6Al-4V produced by rapid-layer manufacturing', authors: ['Murr, L.E.', 'Gaytan, S.M.', 'Ramirez, D.A.'], year: 2012, doi: '10.1016/j.actamat.2012.01.031', source_type: 'journal', journal: 'Acta Materialia', record_count: 12 },
  { id: 'src-002', title: 'Residual stresses in selective laser sintered and hot isostatically pressed Ti-6Al-4V', authors: ['Leuders, S.', 'Thöne, M.', 'Riemer, A.'], year: 2013, doi: '10.1016/j.actamat.2012.12.001', source_type: 'journal', journal: 'Acta Materialia', record_count: 8 },
  { id: 'src-003', title: 'Microstructures and mechanical properties of Ti6Al4V parts fabricated by selective laser melting and electron beam melting', authors: ['Rafi, H.K.', 'Karthik, N.V.', 'Gong, H.'], year: 2013, doi: '10.1007/s11661-013-1845-4', source_type: 'journal', journal: 'Metallurgical and Materials Transactions A', record_count: 10 },
  { id: 'src-004', title: 'Porosity of 3D printed titanium implants: predictable porosity enables personalized design', authors: ['Xu, W.', 'Sun, S.', 'Elambasseril, J.'], year: 2015, doi: '10.1002/adma.201500055', source_type: 'journal', journal: 'Advanced Materials', record_count: 6 },
  { id: 'src-005', title: 'Microstructural and tensile behavior of Ti-6Al-4V alloy produced by SLM', authors: ['Zhang, X.Y.', 'Fang, G.', 'Leeflang, S.'], year: 2017, doi: '10.1016/j.msea.2017.05.025', source_type: 'journal', journal: 'Materials Science and Engineering A', record_count: 9 },
  { id: 'src-006', title: 'Process parameter influence on mechanical properties in laser sintering of PA12', authors: ['Kok, Y.', 'Tan, X.P.', 'Wang, P.'], year: 2018, doi: '10.1016/j.matdes.2018.02.065', source_type: 'journal', journal: 'Materials and Design', record_count: 11 },
  { id: 'src-007', title: 'Defect-structure anisotropy of selective laser-melted Ti6Al4V alloy', authors: ['Benedetti, M.', 'Torresani, E.', 'Leoni, M.'], year: 2017, doi: '10.1016/j.ijfatigue.2016.09.012', source_type: 'journal', journal: 'International Journal of Fatigue', record_count: 7 },
  { id: 'src-008', title: 'Optimization of selective laser melting process parameters for Ti6Al4V based on response surface methodology', authors: ['Kasperovich, G.', 'Haubrich, J.', 'Gussone, J.'], year: 2016, doi: '10.1016/j.msea.2016.01.013', source_type: 'journal', journal: 'Materials Science and Engineering A', record_count: 8 },
];
