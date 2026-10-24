# 1. Update config_loader.py
with open('road_surveillance/config/config_loader.py', 'r', encoding='utf-8') as f:
    cfg_code = f.read()

cfg_code = cfg_code.replace(
    'input_resolution: int = 640',
    'input_resolution: Union[int, str] = "native"'
)
with open('road_surveillance/config/config_loader.py', 'w', encoding='utf-8') as f:
    f.write(cfg_code)
print('Updated config_loader.py')

# 2. Update yolo_detector.py
with open('road_surveillance/detection/yolo_detector.py', 'r', encoding='utf-8') as f:
    det_code = f.read()

det_code = det_code.replace(
    'input_resolution: int = 640,',
    'input_resolution: Union[int, str] = "native",'
)
det_code = det_code.replace(
    '        # Run inference\n        results = self.model.predict(\n            source=img,\n            conf=active_conf,\n            imgsz=self.input_resolution,',
    '        # Native resolution handling\n        det_imgsz = self.input_resolution\n        if det_imgsz is None or str(det_imgsz).lower() in ("native", "none", "0"):\n            ih, iw = img.shape[:2]\n            det_imgsz = (int(np.ceil(ih / 32.0) * 32), int(np.ceil(iw / 32.0) * 32))\n        elif isinstance(det_imgsz, str) and det_imgsz.isdigit():\n            det_imgsz = int(det_imgsz)\n\n        # Run inference\n        results = self.model.predict(\n            source=img,\n            conf=active_conf,\n            imgsz=det_imgsz,'
)
with open('road_surveillance/detection/yolo_detector.py', 'w', encoding='utf-8') as f:
    f.write(det_code)
print('Updated yolo_detector.py')

# 3. Update default_config.yaml
with open('road_surveillance/config/default_config.yaml', 'r', encoding='utf-8') as f:
    yaml_code = f.read()

yaml_code = yaml_code.replace(
    '  input_resolution: 640',
    '  input_resolution: "native"'
)
with open('road_surveillance/config/default_config.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_code)
print('Updated default_config.yaml')

# 4. Update run_pipeline.py
with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    run_code = f.read()

run_code = run_code.replace(
    '    output_json: Optional[str] = None,\n) -> Dict[str, Any]:',
    '    output_json: Optional[str] = None,\n    imgsz: Optional[Union[int, str]] = None,\n) -> Dict[str, Any]:'
)

old_yolo_call = """        # 1. Primary YOLO 640p Pass
        yolo_res = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]"""

new_yolo_call = """        # 1. Primary YOLO Native Feed Inference (preserves native video feed details)
        if imgsz is not None and str(imgsz).lower() not in ("native", "none", "0"):
            det_imgsz = int(imgsz) if str(imgsz).isdigit() else 640
        elif cfg.detection.input_resolution not in (None, "native", "none", "0"):
            det_imgsz = int(cfg.detection.input_resolution) if str(cfg.detection.input_resolution).isdigit() else 640
        else:
            native_stride = 32
            det_imgsz = (
                int(np.ceil(h / float(native_stride)) * native_stride),
                int(np.ceil(w / float(native_stride)) * native_stride),
            )

        yolo_res = yolo_model.predict(source=proc_img, imgsz=det_imgsz, conf=0.02, device="cpu", verbose=False)[0]"""

run_code = run_code.replace(old_yolo_call, new_yolo_call)

run_code = run_code.replace(
    '    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")\n    return parser.parse_args()',
    '    parser.add_argument("--imgsz", "--resolution", type=str, default="native", help="Detector inference resolution (default: \'native\' for unscaled feed, or integer e.g. 640)")\n    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")\n    return parser.parse_args()'
)

run_code = run_code.replace(
    '        output_video=args.output_video,\n        output_json=args.output,\n    )',
    '        output_video=args.output_video,\n        output_json=args.output,\n        imgsz=args.imgsz,\n    )'
)

with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(run_code)
print('Updated run_pipeline.py')
