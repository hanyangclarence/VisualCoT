# Run with Chain of Thought & Verify
```bash
python run_scenegraph.py --apikey "xxx" --caption_type vinvl_ocr --n_shot 2 --n_ensemble 1 --round 2 --iterative_strategy caption --engine chat --chain_of_thoughts --with_clip_verify --device cpu
```

# Run with Chain of Thought withoug Verify
```bash
python run_scenegraph.py --apikey "xxx" --caption_type vinvl_ocr --n_shot 2 --n_ensemble 1 --round 2 --iterative_strategy caption --engine chat --device cpu
```

# Run without Chain of Thought
```bash
python run_scenegraph.py --apikey "xxx" --caption_type vinvl_ocr --n_shot 2 --n_ensemble 1 --round 2 --iterative_strategy caption --engine chat --device cpu --all_regional_captions
```

# Run test on Open-EQA, no verify
```bash
python test_openeqa.py --apikey "xxx" --n_shot 8 --n_ensemble 1 --round 5 --engine chat --device cpu
```