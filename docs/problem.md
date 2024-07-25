# Common problems and solutions

## 1. Unknown scheme for proxy URL URL('socks://127.0.0.1:1081/')

Solution:

```
unset all_proxy && unset ALL_PROXY
```

## 2. libcupti.so.11.7 not found

Solution:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/extras/CUPTI/lib64
```

## 3. ImportError: libnccl.so.2: cannot open shared object file: No such file or directory

Solution:

```
poetry add nvidia-nccl-cu11
```

## 4. Attempting to unscale FP16 gradients.

```
TrainingArguments(fp16=True) and from_pretrained(torch_dtype=torch.float16) cannot be reused
```

## 5. mpi4py cannot be installed

```
sudo apt-get install -y libopenmpi-dev
```

## 6. ImportError: libnccl.so.2: cannot open shared object file: No such file or directory

```
sudo apt install libnccl2
```

## 7. poetry install 'Link' object has no attribute 'name'

```
rm -rf $HOME/.cache/pypoetry/artifacts/*
```

## 8. ./aten/src/ATen/native/cuda/Indexing.cu:1146: indexSelectLargeIndex: block: [54,0,0], thread: [32,0,0] Assertion `srcIndex < srcSelectDimSize` failed.

The data length exceeds the input length of the model. You need to set max_seq_len or MAX_SEQ_LEN.

## 9. ModuleNotFoundError: No module named 'transformers_modules.baichuan_7b_sft_v0' appears when loading the model locally

Because the model address is
```
/media/zjin/Data/dataset/model/trained_model/baichuan_7b_sft_v0.1 contains . , it needs to be removed
and changed to /media/zjin/Data/dataset/model/trained_model/baichuan_7b_sft_v01
```

## 10. chatglm2: Input length of input_ids is 2597, but `max_length` is set to 2574. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.

```
When the input of the glm model is greater than the output, an error will be reported, and max_new_tokens needs to be set larger
```