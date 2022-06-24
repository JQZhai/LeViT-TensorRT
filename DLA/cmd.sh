trtexec --onnx=../LeViT-128S.onnx  \
--saveEngine=Levit_fp16_DLA.engine --fp16 --exportProfile=Levit_fp16_DLA.json \
--useDLACore=0 --allowGPUFallback --useSpinWait --separateProfileRun > Levit_fp16_DLA.log

--minShapes=input_0:1*3*224*224 \
--optShapes=input_0:1*3*224*224 --maxShapes=input_0:1*3*224*22