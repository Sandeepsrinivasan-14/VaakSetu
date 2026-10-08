## STT & TTS MODEL TRAINING

## Technical Exercise – Project Requirements

## 1. Objective

Select and implement open-source, free, and trainable/fine-tunable STT and TTS models. The main objective is

to make the model learn from its incorrect outputs through user corrections.

## 2. Functional Requirements

Select suitable free, open-source and trainable/fine-tunable STT and TTS models. Document the selected models, their capabilities and the reason for selecting them.

The models should support and correctly handle relevant words, sentences, names, numbers, dates, alphanumeric values, symbols, special characters and Unicode terms/characters such as α (alpha), β (beta),

etc., wherever supported by the selected model.

Support the required languages, particularly the target Indian languages. Clearly document which languages are supported and tested.

Identify incorrect STT transcriptions or TTS pronunciations and provide a mechanism for a human to give the correct output. The correction should become part of the training data.

Use the collected corrections to train or fine-tune the actual model/adapter. The improvement must come from model-level learning and should not depend only on hardcoded replacements, dictionaries, lookup tables or application-level rules.

Training and correction data must be stored systematically and remain available for future training. Previously

collected data must not be unintentionally lost when new training is performed.

The training approach should allow new correction data to be added while retaining previously learned useful

behaviour. Demonstrate that additional training does not unnecessarily remove earlier learning.

Save the trained model/adapter together with the required configuration, tokenizer/processor and supporting files.

The trained model must be loadable and reusable in another supported environment.

## 1. Model Selection

## 2. Input and Output Coverage

## 3. Multilingual Support

## 4. Error Correction and Learning

## 5. Model-Level Training

## 6. Training Data Persistence

## 7. Continued Learning

## 8. Model Reusability

## 3. Technical Documentation

- Base STT/TTS model used.

- Training/fine-tuning approach.

- Trainable and frozen layers/parameters.

- Training dataset and data format.

- Important training parameters such as learning rate, epochs and batch size.

- Model/adapter storage and versioning approach.

- How new correction data is incorporated into future training.


## 4. Expected Demonstration

## Base Model → Incorrect Output → Human Correction → Training Data → Fine-Tuning → Updated Model → Retest → Improved Output

The demonstration must include before-training and after-training results and should show examples involving

names, numbers, symbols/Unicode terms and supported languages.

## 5. Expected Deliverables

- Working STT model and TTS model.

- Training/fine-tuning code.

- Training and correction dataset.

- Saved trained model/adapter and required supporting files.

- Technical documentation.

- Before/after training results demonstrating improvement.

- Short explanation/demo of how the model can be further trained using newly collected corrections.
