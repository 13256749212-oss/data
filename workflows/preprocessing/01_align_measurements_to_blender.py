#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将原始 Cellular-Pro 经纬度数据转换到 BlenderGIS 局部坐标并采样 DEM。

默认流程：WGS84(EPSG:4326) -> Web Mercator(EPSG:3857) -> 减去 BlenderGIS 场景原点。
地面高程直接在 assets/ground.ply 的三角网格上插值，接收高度为 ground_z+1.5 m。
输入文件不会被覆盖；每个输出文件添加 _with_blender_xyz 后缀。
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
from pyproj import Transformer
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parents[2]

def _find_column(columns, aliases):
    norm = {str(c).strip().lower().replace(' ', '').replace('_',''): c for c in columns}
    for a in aliases:
        k = a.strip().lower().replace(' ', '').replace('_','')
        if k in norm: return norm[k]
    raise KeyError(f"找不到字段，候选={aliases}，实际字段={list(columns)}")

def load_ground(path: Path):
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices)<3 or len(mesh.faces)<1:
        raise ValueError(f"ground.ply不是有效三角网格: {path}")
    v=np.asarray(mesh.vertices,float); f=np.asarray(mesh.faces,int)
    if f.ndim!=2 or f.shape[1]!=3: raise ValueError('ground.ply必须为三角面')
    tri=mtri.Triangulation(v[:,0],v[:,1],f)
    interp=mtri.LinearTriInterpolator(tri,v[:,2])
    return mesh, interp

def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ('utf-8-sig','utf-8','gb18030'):
        try: return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError: pass
    raise UnicodeDecodeError('csv','',0,1,f'无法识别编码: {path}')

def parse_args():
    p=argparse.ArgumentParser(description='Cellular-Pro坐标转换、BlenderGIS对齐和DEM+1.5m高度提取')
    p.add_argument('--input-dir', default=str(ROOT/'data/raw_measurements'))
    p.add_argument('--output-dir', default=str(ROOT/'data/aligned_measurements'))
    p.add_argument('--ground', default=str(ROOT/'assets/ground.ply'))
    p.add_argument('--alignment-config', default=str(ROOT/'config/coordinate_alignment.json'))
    p.add_argument('--glob', default='*.csv')
    p.add_argument('--force', action='store_true')
    return p.parse_args()

def main():
    a=parse_args(); inp=Path(a.input_dir); out=Path(a.output_dir); ground=Path(a.ground); cfgp=Path(a.alignment_config)
    cfg=json.loads(cfgp.read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=True)
    files=[p for p in sorted(inp.glob(a.glob)) if p.is_file() and '_with_blender_xyz' not in p.stem]
    if not files: raise FileNotFoundError(f'在 {inp} 中没有找到原始CSV')
    mesh, interp=load_ground(ground)
    transformer=Transformer.from_crs(cfg.get('source_crs','EPSG:4326'),cfg.get('projected_crs','EPSG:3857'),always_xy=True)
    ox=float(cfg['scene_origin_projected_x']); oy=float(cfg['scene_origin_projected_y'])
    sx=float(cfg.get('blender_x_scale',1.0)); sy=float(cfg.get('blender_y_scale',1.0))
    xoff=float(cfg.get('blender_x_offset_m',0.0)); yoff=float(cfg.get('blender_y_offset_m',0.0))
    rxh=float(cfg.get('receiver_height_agl_m',1.5))
    all_parts=[]; reports=[]
    for path in files:
        target=out/f'{path.stem}_with_blender_xyz.csv'
        if target.exists() and not a.force:
            print('[SKIP]',target); continue
        df=read_csv_robust(path)
        lonc=_find_column(df.columns,['LONGITUDE','longitude','lon']); latc=_find_column(df.columns,['LATITUDE','latitude','lat'])
        lon=pd.to_numeric(df[lonc],errors='coerce').to_numpy(float); lat=pd.to_numeric(df[latc],errors='coerce').to_numpy(float)
        valid=np.isfinite(lon)&np.isfinite(lat)&(np.abs(lon)<=180)&(np.abs(lat)<=90)&~((lon==0)&(lat==0))
        px=np.full(len(df),np.nan); py=np.full(len(df),np.nan)
        if valid.any(): px[valid],py[valid]=transformer.transform(lon[valid],lat[valid])
        bx=(px-ox)*sx+xoff; by=(py-oy)*sy+yoff
        gz=np.full(len(df),np.nan)
        q=valid&np.isfinite(bx)&np.isfinite(by)
        if q.any():
            z=interp(bx[q],by[q]); z=np.ma.asarray(z); gz[q]=z.filled(np.nan)
        hit=np.isfinite(gz)
        df['epsg3857_x']=px; df['epsg3857_y']=py; df['blender_x']=bx; df['blender_y']=by
        df['ground_z_m']=gz; df['receiver_z_m']=gz+rxh; df['dem_hit']=hit.astype(int); df['dem_object']='ground'
        df.to_csv(target,index=False,encoding='utf-8-sig')
        part=df.copy(); part.insert(0,'source_row',np.arange(2,len(df)+2)); part.insert(0,'source_file',path.name); all_parts.append(part)
        reports.append({'source_file':path.name,'row_count':int(len(df)),'valid_lonlat_count':int(valid.sum()),'dem_hit_count':int(hit.sum()),'dem_hit_rate':float(hit.mean())})
        print('[OK]',path.name,'->',target.name,'DEM命中',int(hit.sum()),'/',len(df))
    if all_parts:
        pd.concat(all_parts,ignore_index=True).to_csv(out/'all_measurements_blender_xyz.csv',index=False,encoding='utf-8-sig')
    report={'alignment_config':cfg,'ground_bounds':np.asarray(mesh.bounds,float).tolist(),'files':reports}
    (out/'alignment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完成，输出目录:',out)
if __name__=='__main__': main()
