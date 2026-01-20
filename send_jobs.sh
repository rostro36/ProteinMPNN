#!/bin/bash

GPU="a5000"

OMEGAS=("0") # "10")
MATRICES=("blosum")
BETAS=("0.5") #  "0.25" "0.1")
LAMBDAS=("5")

#PRETRAINING_ID=$(sbatch --parsable ProteinMPNN_pretraining.slurm)

# Loop through array
for OMEGA in "${OMEGAS[@]}"; do 
  for MATRIX in "${MATRICES[@]}"; do
    for BETA in "${BETAS[@]}"; do
      for LAMBDA in "${LAMBDAS[@]}"; do
        # Generate files
        sed \
          -e "s/ööööö/${BETA}/g" \
          -e "s/äääää/${MATRIX}/g" \
          -e "s/üüüüü/${GPU}/g" \
          -e "s/ééééé/${OMEGA}/g" \
          -e "s/èèèèè/${LAMBDA}/g" \
          "ProteinMPNN_finetune_template.txt" > "ProteinMPNN_finetune_longer_${MATRIX}_${BETA}_${OMEGA}_${LAMBDA}.slurm"
        sed \
          -e "s/ööööö/${BETA}/g" \
          -e "s/äääää/${MATRIX}/g" \
          -e "s/üüüüü/${GPU}/g" \
          -e "s/ééééé/${OMEGA}/g" \
          -e "s/èèèèè/${LAMBDA}/g" \
          "ProteinMPNN_inference_template.txt" > "ProteinMPNN_inference_longer_${MATRIX}_${BETA}_${OMEGA}_${LAMBDA}.slurm"
        # send files
        ID=$(sbatch --parsable ProteinMPNN_finetune_longer_${MATRIX}_${BETA}_${OMEGA}_${LAMBDA}.slurm)
        # --dependency=afterany:${PRETRAINING_ID}
        sbatch --parsable --dependency=afterany:${ID} ProteinMPNN_inference_longer_${MATRIX}_${BETA}_${OMEGA}_${LAMBDA}.slurm
      done
    done
  done
done