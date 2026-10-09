import pandas as pd

df = pd.read_csv(r'C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv')
vids = [
    'nAXTthLfgtI_00', 'vxZ47DoPNY8_00', 'VozrseCzhQc_00', 'UarP8qU1S-c_00', 'UarP8qU1S-c_01',
    'Z4kg2Ev3vhk_00', '82sQ7N-aHOw_00', 'bJMc3sNP2AU_00', 'ftk_fSc4haM_00', 'BgdKGnFqz0I_00',
    'aPsS_HaokoU_00', 'unS0-TLF1ao_00', 'iA_u73BG4ZM_00'
]
sub = df[df['path'].apply(lambda p: any(v in str(p) for v in vids))]
for _, r in sub.iterrows():
    p = r['path']
    t = r['type']
    roll = r['rollover']
    af = r['accident_frame']
    box = (r['x1'], r['y1'], r['x2'], r['y2'])
    sin = r['split_in_distribution']
    sgeo = r['split_geo_aware']
    print(f"{p} | type={t} | roll={roll} | acc_f={af} | bbox={box} | in={sin} | geo={sgeo}")
