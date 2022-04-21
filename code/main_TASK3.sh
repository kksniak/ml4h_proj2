
# Create datasets
echo 'Preparing datasets...'
python utils.py

# Load models and evaluate them on a small test set
python -m models.bert --params results/Bio_ClinicalBERT_untuned/params.txt --checkpoint results/Bio_ClinicalBERT_untuned/checkpoint/checkpoint --dataset debug --skip-training
python -m models.bert --params results/Bio_ClinicalBERT_tuned/params.txt --checkpoint results/Bio_ClinicalBERT_tuned/checkpoint/checkpoint --dataset debug --skip-training
python -m models.bert --params results/SapBERT_tuned/params.txt --checkpoint results/SapBERT_tuned/checkpoint/checkpoint --dataset debug --skip-training

# Uncomment to evaluate on the full test set (takes a few hours on CPU, a few minutes on GPU)
# python -m models.bert --params results/Bio_ClinicalBERT_untuned/params.txt --checkpoint results/Bio_ClinicalBERT_untuned/checkpoint/checkpoint --dataset small_balanced --skip-training
# python -m models.bert --params results/Bio_ClinicalBERT_tuned/params.txt --checkpoint results/Bio_ClinicalBERT_tuned/checkpoint/checkpoint --dataset small_balanced --skip-training
# python -m models.bert --params results/SapBERT_tuned/params.txt --checkpoint results/SapBERT_tuned/checkpoint/checkpoint --dataset small_balanced --skip-training

# Uncomment to train models from scratch and evaluate (takes a few hours on GPU)
# python -m models.bert --params results/Bio_ClinicalBERT_tuned/params.txt
# python -m models.bert --params results/Bio_ClinicalBERT_untuned/params.txt
# python -m models.bert --params results/SapBERT_tuned/params.txt