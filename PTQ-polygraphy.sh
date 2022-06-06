polygraphy run LeViT-128S.onnx --onnxrt --trt --workspace 28G \
--save-engine=LeViT-128S.onnx_fp16.plan --atol 1e-3 --rtol 1e-3 --verbose \
--input-shape 'input_0:[1,3,224,224]' \
--onnx-outputs mark all  --trt-outputs mark all \
--fp16 


polygraphy debug precision ../LeViT-128S.onnx \
-v --fp16 --workspace 28G --no-remove-intermediate --log-file ./log_file.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
-p fp32 --mode bisect --dir forward --show-output \
--artifacts ./polygraphy_debug.engine --art-dir ./art-dir \
--check \
polygraphy run polygraphy_debug.engine \
--trt --load-outputs onnx_res.json --load-inputs onnx_input.json \
--abs 1e-2 -v --rel 1e-2


polygraphy run ../LeViT-128S.onnx \
--onnxrt -v --workspace=28G --fp16 \
--input-shapes 'input_0:[1,3,224,224]' --onnx-outputs mark all \
--save-inputs onnx_input.json --save-outputs onnx_res.json


polygraphy convert -v --model-type onnx --input-shapes 'input_0:[1,3,224,224]' \
--shape-inference --seed 7 --load-inputs ../fp16/onnx_input.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
--int8 --workspace 30G --calibration-cache ./cal_trt.bin -o LeViT-128S.plan \
../LeViT-128S.onnx

polygraphy run LeViT-128S.plan \
--trt -v --workspace=28G  --model-type engine --int8 \
--input-shapes 'input_0:[1,3,224,224]' --calibration-cache ./cal_trt.bin \
--load-inputs ../fp16/onnx_input.json --load-outputs ../fp16/onnx_res.json \
--abs 1e-2 --rel 1e-2 

polygraphy debug precision ../LeViT-128S.onnx \
-v --int8 --workspace 28G --no-remove-intermediate --log-file ./log_file.json \
--trt-min-shapes 'input_0:[1,3,224,224]' \
--trt-opt-shapes 'input_0:[1,3,224,224]' \
--trt-max-shapes 'input_0:[1,3,224,224]' \
-p fp16 --mode bisect --dir forward --show-output --calibration-cache ./cal_trt.bin \
--artifacts ./polygraphy_debug.engine --art-dir ./art-dir \
--check \
polygraphy run polygraphy_debug.engine \
--trt --load-outputs ../fp16/onnx_res.json --load-inputs ../fp16/onnx_input.json \
--abs 1e-2 -v --rel 1e-2