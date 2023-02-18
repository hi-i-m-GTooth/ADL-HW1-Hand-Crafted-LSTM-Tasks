# NTU-ADL-HW1-R11922A16-柳宇澤

## Prediction

### Download models
```
./download.sh
```
### Predicit
```shell
# {TEST}: path to the testing file.
# {OUTPUT}: path to the output predictions.

# Intent classification
./intent_cls.sh {TEST} {OUTPUT}
# Slot tagging
./slot_tag {TEST} {OUTPUT}
```

## Reproduce

### Train
```bash
./run_train_intent.sh
./run_train_slot.sh
```

### Prediction on Test
```bash
./run_test_intent.sh
./run_test_slot.sh
```
Then check outputs, `pred.intent.csv` & `pred.slot.csv`



