#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-point Rank-Sector Geometric Localization (RSGL-v1.11).

This workflow is intentionally independent of the previous DP-PGRSL / DP-PPRSL
solvers.  It estimates only the transmitter XY coordinates from randomly sampled
receiver locations.  The objective combines:
  1) same-location inter-sector RSRP differences (path-loss / power cancel);
  2) fixed external sector-direction geometry;
  3) RSRP-distance ordering consistency; and
  4) a low-dimensional robust path-loss fit used only to score a candidate XY.

No true transmitter coordinate is used before final evaluation.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import legacy_pgrmsbil as common

ALGORITHM_NAME = "Rank-Sector Geometric Localization (RSGL-v1.11)"
ROOT = Path(__file__).resolve().parents[2]
MIN_RSRP_DBM = -120.0
MAX_RSRP_DBM = -40.0
HORIZONTAL_3DB_BEAMWIDTH_DEG = 65.0
HORIZONTAL_MAX_ATTENUATION_DB = 30.0
VERTICAL_SEPARATION_M = 28.5
DEFAULT_BOUNDS = common.DEFAULT_BOUNDS


def parse_args():
    p=argparse.ArgumentParser(description="RSGL-v1.11 random sparse base-station localization")
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--measurements", type=Path, default=None)
    p.add_argument("--directions", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--points-per-station", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=20260805)
    p.add_argument("--direction-prior-mode", choices=["fixed","off","soft"], default="fixed")
    p.add_argument("--bootstrap", type=int, default=0, help="Compatibility option; Monte-Carlo trials are the uncertainty experiment")
    p.add_argument("--de-maxiter", type=int, default=100, help="Compatibility option, unused by RSGL")
    p.add_argument("--de-popsize", type=int, default=10, help="Compatibility option, unused by RSGL")
    p.add_argument("--station-ids", default="all")
    p.add_argument("--x-min", type=float, default=DEFAULT_BOUNDS[0]); p.add_argument("--x-max", type=float, default=DEFAULT_BOUNDS[1])
    p.add_argument("--y-min", type=float, default=DEFAULT_BOUNDS[2]); p.add_argument("--y-max", type=float, default=DEFAULT_BOUNDS[3])
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-per-station-figures", action="store_true")
    return p.parse_args()


def resolve_direction_csv(project_root:Path, explicit:Optional[Path])->Optional[Path]:
    if explicit is not None:
        p=explicit.expanduser().resolve(); return p if p.is_file() else None
    cands=[
        project_root/"outputs/parameter_calibration/estimated_initial_directions_27stations.csv",
        project_root/"config/estimated_initial_directions_27stations.csv",
    ]
    return next((p.resolve() for p in cands if p.is_file()), None)


def load_directions(path:Optional[Path])->pd.DataFrame:
    if path is None: return pd.DataFrame()
    d=pd.read_csv(path, encoding="utf-8-sig")
    if "station_id" not in d or "base_alpha_rad" not in d: return pd.DataFrame()
    d["station_id"]=pd.to_numeric(d["station_id"], errors="coerce").astype("Int64")
    return d.dropna(subset=["station_id"]).set_index("station_id")


def random_points(points:pd.DataFrame, k:int, seed:int, omni:bool)->pd.DataFrame:
    cols=["rsrp_s1","rsrp_s2","rsrp_s3"]
    valid=points[cols].notna().any(axis=1)
    pool=points.loc[valid].copy().reset_index(drop=True)
    if len(pool)<k:
        raise ValueError(f"定位有效实测位置仅{len(pool)}个，少于要求{k}个")
    rng=np.random.default_rng(int(seed))
    idx=rng.choice(len(pool), size=int(k), replace=False)
    out=pool.iloc[idx].copy().reset_index(drop=True)
    out["selection_rank"]=np.arange(1,len(out)+1)
    out["observed_sector_count"]=out[cols].notna().sum(axis=1)
    return out


def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi


def sector_gain_db(offset_rad:np.ndarray)->np.ndarray:
    deg=np.degrees(np.abs(wrap(offset_rad)))
    return -np.minimum(12.0*(deg/HORIZONTAL_3DB_BEAMWIDTH_DEG)**2, HORIZONTAL_MAX_ATTENUATION_DB)


def sector_angles(direction_row:Optional[pd.Series], mode:str)->tuple[np.ndarray,bool]:
    if mode=="off" or direction_row is None or not np.isfinite(pd.to_numeric(direction_row.get("base_alpha_rad",np.nan), errors="coerce")):
        return np.asarray([0.0, 2*np.pi/3, -2*np.pi/3]), False
    alpha=float(direction_row["base_alpha_rad"])
    order=str(direction_row.get("selected_sector_order",""))
    sign=1 if "plus120" in order else -1
    return alpha+np.asarray([0.0, sign*2*np.pi/3, -sign*2*np.pi/3]), True


def weighted_line_intersection(selected:pd.DataFrame, angles:np.ndarray)->np.ndarray:
    xy=selected[["x_m","y_m"]].to_numpy(float)
    rss=selected[["rsrp_s1","rsrp_s2","rsrp_s3"]].to_numpy(float)
    A=[]; b=[]; w=[]
    for j in range(3):
        vals=rss[:,j]; mask=np.isfinite(vals)
        if not mask.any(): continue
        vmax=float(np.nanmax(vals[mask]))
        for p,r in zip(xy[mask], vals[mask]):
            u=np.asarray([math.cos(angles[j]), math.sin(angles[j])])
            n=np.asarray([-u[1],u[0]])
            A.append(n); b.append(float(n@p)); w.append(float(np.exp((r-vmax)/8.0)))
    if len(A)<2: return np.mean(xy,axis=0)
    A=np.asarray(A); b=np.asarray(b); w=np.sqrt(np.clip(np.asarray(w),0.05,1.0))
    try: return np.linalg.lstsq(A*w[:,None], b*w, rcond=None)[0]
    except Exception: return np.mean(xy,axis=0)


def robust_pathloss_score(tx:np.ndarray, selected:pd.DataFrame, angles:np.ndarray, omni:bool)->tuple[float,float,float]:
    xy=selected[["x_m","y_m"]].to_numpy(float)
    rss=selected[["rsrp_s1","rsrp_s2","rsrp_s3"]].to_numpy(float)
    rows=[]; y=[]
    for i,p in enumerate(xy):
        d3=float(np.sqrt(np.sum((p-tx)**2)+VERTICAL_SEPARATION_M**2))
        bearing=math.atan2(p[1]-tx[1],p[0]-tx[0])
        for j in range(3):
            if not np.isfinite(rss[i,j]): continue
            gain=0.0 if omni else float(sector_gain_db(np.asarray([bearing-angles[j]]))[0])
            # Per-sector intercept plus one shared path-loss slope.
            row=[0.0,0.0,0.0, -10.0*math.log10(max(d3,1.0))]
            row[j]=1.0
            rows.append(row); y.append(float(rss[i,j]-gain))
    if len(rows)<4: return 30.0, np.nan, np.nan
    X=np.asarray(rows,float); y=np.asarray(y,float)
    # IRLS-Huber. Slope column coefficient is n directly.
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(4):
        pred=X@coef; r=y-pred
        scale=max(1.4826*np.median(np.abs(r-np.median(r))),2.0)
        u=np.abs(r)/(1.5*scale); ww=np.ones_like(u); m=u>1; ww[m]=1/u[m]
        coef=np.linalg.lstsq(X*np.sqrt(ww)[:,None], y*np.sqrt(ww), rcond=None)[0]
        coef[3]=np.clip(coef[3],1.5,5.5)
    residual=y-X@coef
    score=float(np.mean(np.log1p((residual/5.0)**2)))
    rmse=float(np.sqrt(np.mean(residual**2)))
    return score,float(coef[3]),rmse


def pair_sector_score(tx:np.ndarray, selected:pd.DataFrame, angles:np.ndarray, omni:bool)->float:
    if omni: return 0.0
    xy=selected[["x_m","y_m"]].to_numpy(float)
    rss=selected[["rsrp_s1","rsrp_s2","rsrp_s3"]].to_numpy(float)
    losses=[]
    for i,p in enumerate(xy):
        bearing=math.atan2(p[1]-tx[1],p[0]-tx[0])
        gains=np.asarray([sector_gain_db(np.asarray([bearing-a]))[0] for a in angles])
        for a in range(3):
            for b in range(a+1,3):
                if np.isfinite(rss[i,a]) and np.isfinite(rss[i,b]):
                    res=(rss[i,a]-rss[i,b])-(gains[a]-gains[b])
                    losses.append(math.log1p((res/5.0)**2))
    return float(np.mean(losses)) if losses else 0.0


def rank_score(tx:np.ndarray, selected:pd.DataFrame)->float:
    xy=selected[["x_m","y_m"]].to_numpy(float)
    rss=selected[["rsrp_s1","rsrp_s2","rsrp_s3"]].to_numpy(float)
    d=np.sqrt(np.sum((xy-tx[None,:])**2,axis=1)+VERTICAL_SEPARATION_M**2)
    penalties=[]
    for j in range(3):
        idx=np.flatnonzero(np.isfinite(rss[:,j]))
        for aa in range(len(idx)):
            for bb in range(aa):
                i,k=idx[aa],idx[bb]; dr=rss[i,j]-rss[k,j]
                if abs(dr)<3.0: continue
                # Stronger signal should normally be closer. Penalize reversals only.
                signed=(d[i]-d[k])*np.sign(dr)
                if signed>0: penalties.append(min(float(signed)/100.0,3.0)**2)
    return float(np.mean(penalties)) if penalties else 0.0


def candidate_objective(tx:np.ndarray, selected:pd.DataFrame, angles:np.ndarray, omni:bool, center:np.ndarray, span:float)->float:
    path,_,_=robust_pathloss_score(tx,selected,angles,omni)
    pair=pair_sector_score(tx,selected,angles,omni)
    rank=rank_score(tx,selected)
    weak=0.01*(float(np.linalg.norm(tx-center))/max(span,100.0))**2
    return 0.58*path+0.32*pair+0.10*rank+weak


def solve(selected:pd.DataFrame, angles:np.ndarray, omni:bool, bounds:Sequence[float])->dict:
    xy=selected[["x_m","y_m"]].to_numpy(float)
    # Fully different low-dimensional solver: deterministic multiresolution grid + Powell.
    if omni:
        strongest=selected[["rsrp_s1","rsrp_s2","rsrp_s3"]].max(axis=1).to_numpy(float)
        ww=np.exp((strongest-np.nanmax(strongest))/7.0); center=np.average(xy,axis=0,weights=ww)
    else:
        center=weighted_line_intersection(selected,angles)
        if not np.isfinite(center).all(): center=np.mean(xy,axis=0)
    spread=max(common.point_spread(selected),100.0)
    x0,x1,y0,y1=map(float,bounds)
    lo=np.maximum([x0,y0], np.min(xy,axis=0)-450.0); hi=np.minimum([x1,y1], np.max(xy,axis=0)+450.0)
    # Ensure direction-based center is represented even if it is outside sample bounding box.
    lo=np.minimum(lo, np.maximum([x0,y0],center-300)); hi=np.maximum(hi,np.minimum([x1,y1],center+300))
    best=np.clip(center,lo,hi); bestval=candidate_objective(best,selected,angles,omni,center,spread)
    for step in (120.0,45.0,15.0):
        radius=3*step
        xs=np.arange(max(lo[0],best[0]-radius), min(hi[0],best[0]+radius)+0.1, step)
        ys=np.arange(max(lo[1],best[1]-radius), min(hi[1],best[1]+radius)+0.1, step)
        for xx in xs:
            for yy in ys:
                v=candidate_objective(np.asarray([xx,yy]),selected,angles,omni,center,spread)
                if v<bestval: bestval=v; best=np.asarray([xx,yy])
    res=minimize(lambda z:candidate_objective(np.asarray(z,float),selected,angles,omni,center,spread),best,
                 method="Powell",bounds=[(lo[0],hi[0]),(lo[1],hi[1])],options={"maxiter":180,"xtol":0.5,"ftol":1e-4})
    final=np.asarray(res.x if res.success else best,float)
    pscore,n,fitrmse=robust_pathloss_score(final,selected,angles,omni)
    return {"final_xy":final,"objective":float(candidate_objective(final,selected,angles,omni,center,spread)),
            "pathloss_exponent":n,"fit_rmse_db":fitrmse,"initial_xy":center,"solver_mode":"rank_sector_geometric"}


def plot_station(selected, pred, truth, station_id, out):
    fig,ax=plt.subplots(figsize=(6.5,5.4),dpi=180)
    ax.scatter(selected.x_m,selected.y_m,s=38,facecolors="none",edgecolors="black",label="Random measured points")
    ax.scatter(pred[0],pred[1],marker="*",s=170,c="red",edgecolors="black",label="Estimate")
    ax.scatter(truth[0],truth[1],marker="x",s=100,c="black",linewidths=2,label="Ground truth")
    ax.plot([pred[0],truth[0]],[pred[1],truth[1]],"--",lw=1)
    ax.set_aspect("equal",adjustable="datalim"); ax.grid(True,alpha=.25); ax.set_xlabel("Blender X [m]"); ax.set_ylabel("Blender Y [m]")
    ax.set_title(f"Station {station_id} random-point localization"); ax.legend(loc="best",fontsize=8)
    fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)


def main():
    args=parse_args(); project=args.project_root.expanduser().resolve()
    mpath=common.resolve_measurement_csv(project,args.measurements)
    loc,truth=common.load_and_filter(mpath)
    # Enforce requested analysis range everywhere.
    loc=loc[loc["rsrp_dbm"].between(MIN_RSRP_DBM,MAX_RSRP_DBM,inclusive="both")].copy()
    dirs=load_directions(resolve_direction_csv(project,args.directions))
    station_ids=common.parse_station_ids(args.station_ids,sorted(loc.station_id.unique().astype(int)))
    tindex=truth.set_index("station_id")
    outdir=args.output_dir.expanduser().resolve() if args.output_dir else project/"outputs"/f"localization_rsgl_{args.points_per_station}points_seed_{args.random_seed}"
    outdir.mkdir(parents=True,exist_ok=True); per=outdir/"per_station"; per.mkdir(exist_ok=True)
    rows=[]; selected_rows=[]
    for sid in station_ids:
        st=loc[loc.station_id.eq(sid)].copy(); tr=tindex.loc[sid]; omni=bool(int(tr.is_omnidirectional)) or sid==22
        pts=common.point_table(st)
        selected=random_points(pts,args.points_per_station,args.random_seed+sid*7919,omni)
        drow=dirs.loc[sid] if sid in dirs.index else None
        angles,used=sector_angles(drow,args.direction_prior_mode)
        t0=time.time(); sol=solve(selected,angles,omni,(args.x_min,args.x_max,args.y_min,args.y_max)); elapsed=time.time()-t0
        pred=np.asarray(sol["final_xy"],float); true=np.asarray([tr.true_x_m,tr.true_y_m],float); delta=pred-true; err=float(np.linalg.norm(delta))
        obs=common.observations_from_points(selected)
        row={"station_id":sid,"station_label":str(tr.station_label),"antenna_type":str(tr.antenna_type),
             "selected_point_count":len(selected),"observation_count":len(obs),"distinct_pci_count":int(obs.pci.nunique()) if len(obs) else 0,
             "predicted_x_m":pred[0],"predicted_y_m":pred[1],"true_x_m":true[0],"true_y_m":true[1],
             "east_error_m":delta[0],"north_error_m":delta[1],"horizontal_error_m":err,
             "pathloss_exponent":sol["pathloss_exponent"],"selected_fit_rmse_db":sol["fit_rmse_db"],"objective_value":sol["objective"],
             "direction_prior_used":used,"point_spread_m":common.point_spread(selected),"quality_flag":"ok","solver_mode":sol["solver_mode"],"elapsed_s":elapsed,
             "rsrp_min_dbm":MIN_RSRP_DBM,"rsrp_max_dbm":MAX_RSRP_DBM}
        rows.append(row); so=selected.copy(); so.insert(0,"station_id",sid); selected_rows.append(so)
        if not args.skip_figures and not args.skip_per_station_figures: plot_station(selected,pred,true,sid,per/f"station_{sid:02d}_localization.png")
        print(f"Station {sid:02d}: points={len(selected)}, obs={len(obs)}, estimate=({pred[0]:.2f},{pred[1]:.2f}), error={err:.2f} m")
    results=pd.DataFrame(rows).sort_values("station_id"); results.to_csv(outdir/f"localization_results_{len(results)}stations_{args.points_per_station}points.csv",index=False,encoding="utf-8-sig")
    pd.concat(selected_rows,ignore_index=True).to_csv(outdir/f"selected_{args.points_per_station}_random_points_all_stations.csv",index=False,encoding="utf-8-sig")
    e=results.horizontal_error_m.to_numpy(float)
    summary=pd.DataFrame([{"algorithm":ALGORITHM_NAME,"station_count":len(e),"requested_points_per_station":args.points_per_station,
        "mean_error_m":np.mean(e),"median_error_m":np.median(e),"rmse_m":np.sqrt(np.mean(e**2)),"p90_error_m":np.percentile(e,90),"p95_error_m":np.percentile(e,95),"max_error_m":np.max(e),
        "within_50m_percent":np.mean(e<=50)*100,"within_100m_percent":np.mean(e<=100)*100}])
    summary.to_csv(outdir/"localization_accuracy_summary.csv",index=False,encoding="utf-8-sig")
    (outdir/"experiment_metadata.json").write_text(json.dumps({"algorithm":ALGORITHM_NAME,"rsrp_range_dbm":[-120,-40],"random_seed":args.random_seed,"points_per_station":args.points_per_station,"truth_used_only_for_final_evaluation":True},indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); return 0

if __name__=="__main__": raise SystemExit(main())
