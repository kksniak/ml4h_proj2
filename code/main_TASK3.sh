# Run this from the `code` directory

# Create datasets
echo 'Preparing datasets...'
python utils.py

# Load models and evaluate them on a small test set
echo 'Evaluating models on debug test set...'
python -m models.bert --params ../results/Bio_ClinicalBERT_untuned/params.txt --checkpoint models/models_checkpoints/Bio_ClinicalBERT_untuned/checkpoint --dataset debug --skip-training
python -m models.bert --params ../results/Bio_ClinicalBERT_tuned/params.txt --checkpoint models/models_checkpoints/Bio_ClinicalBERT_tuned/checkpoint --dataset debug --skip-training
python -m models.bert --params ../results/SapBERT_tuned/params.txt --checkpoint models/models_checkpoints/SapBERT_tuned/checkpoint --dataset debug --skip-training
python -m models.bert --params ../results/SapBERT_untuned/params.txt --checkpoint models/models_checkpoints/SapBERT_untuned/checkpoint --dataset debug --skip-training

# Uncomment to evaluate on the full test set (takes a few hours on CPU, a few minutes on GPU)
# echo 'Evaluating models on full test set...'
# python -m models.bert --params ../results/Bio_ClinicalBERT_untuned/params.txt --checkpoint models/models_checkpoints/Bio_ClinicalBERT_untuned/checkpoint --dataset small_balanced --skip-training
# python -m models.bert --params ../results/Bio_ClinicalBERT_tuned/params.txt --checkpoint models/models_checkpoints/Bio_ClinicalBERT_tuned/checkpoint --dataset small_balanced --skip-training
# python -m models.bert --params ../results/SapBERT_tuned/params.txt --checkpoint models/models_checkpoints/SapBERT_tuned/checkpoint --dataset small_balanced --skip-training
# python -m models.bert --params ../results/SapBERT_untuned/params.txt --checkpoint models/models_checkpoints/SapBERT_untuned/checkpoint --dataset small_balanced --skip-training

# Uncomment to train models from scratch and evaluate (takes a few hours on GPU)
# echo 'Training models from scratch...'
# python -m models.bert --params ../results/Bio_ClinicalBERT_untuned/params.txt
# python -m models.bert --params ../results/Bio_ClinicalBERT_tuned/params.txt
# python -m models.bert --params ../results/SapBERT_tuned/params.txt
# python -m models.bert --params ../results/SapBERT_untuned/params.txt