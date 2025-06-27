# Model Overview

## Description:

GenMol is a masked diffusion model<sup>1</sup> trained on molecular Sequential Attachment-based Fragment Embedding ([SAFE](https://arxiv.org/abs/2310.10773)) representations<sup>2</sup> for fragment-based molecule generation, which can serve as a generalist model for various drug discovery tasks, including De Novo generation​, linker design​, motif extension​, scaffold decoration/morphing​, hit generation​, and lead optimization.

This model is ready for commercial use.

## License/Terms of Use:

GenMol source code is licensed under Apache 2.0 and the model is licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). By using GenMol, you accept the terms and conditions of this license.

## References:

```bibtex
@misc{sahoo2024simpleeffectivemaskeddiffusion,  
      title={Simple and Effective Masked Diffusion Language Models},   
      author={Subham Sekhar Sahoo and Marianne Arriola and Yair Schiff and Aaron Gokaslan and Edgar Marroquin and Justin T Chiu and Alexander Rush and Volodymyr Kuleshov},  
      year={2024},  
      eprint={2406.07524},  
      archivePrefix={arXiv},  
      primaryClass={cs.CL},  
      url={https://arxiv.org/abs/2406.07524},   
}
```

```bibtex
@misc{noutahi2023gottasafenewframework,  
      title={Gotta be SAFE: A New Framework for Molecular Design},   
      author={Emmanuel Noutahi and Cristian Gabellini and Michael Craig and Jonathan S. C Lim and Prudencio Tossou},  
      year={2023},  
      eprint={2310.10773},  
      archivePrefix={arXiv},  
      primaryClass={cs.LG},  
      url={https://arxiv.org/abs/2310.10773},   
}
```

## Model Architecture:  
**Architecture Type:** Transformer <br>  
**Network Architecture:** BERT <br>

## Input:   
**Input Type(s):** Text (Molecular Sequence), Number (Molecules to generate, SoftMax temperature scaling factor, randomness factor,  diffusion step-size), Enumeration (Scoring method), Binary (Showing unique molecules only) <br>  
**Input Format(s):** Text: String (Sequential Attachment-based Fragment Embedding (SAFE)); Number: Integer, FP32; Enumeration: String (QED, LogP); Binary: Boolean <br>  
**Input Parameters:** 1D <br>  
**Other Properties Related to Input:** Maximum input length is 512 tokens.

## Output:   
**Output Type(s):** Text (List of molecule sequences), Number (List of scores)<br>  
**Output Format:** Text: Array of string (Sequential Attachment-based Fragment Embedding (SAFE)); Number: Array of FP32 (Scores)<br>  
**Output Parameters:** 2D <br>  
**Other Properties Related to Output:** Maximum output length is 512 tokens. <br> 

## Software Integration:  
**Runtime Engine(s):**  
PyTorch >= 2.5.1 <br>

**Supported Hardware Microarchitecture Compatibility:** <br>  
NVIDIA Ampere <br>  
NVIDIA Ada Lovelace <br>  
NVIDIA Hopper <br>  
NVIDIA Grace Hopper <br>

**[Preferred/Supported] Operating System(s):** <br>  
Linux <br>

## Model Version(s):  
GenMol v1.0 <br>

# Training & Evaluation Dataset:

## Training and Testing Dataset:

**Link:** SAFE-GPT [GitHub](https://github.com/datamol-io/safe), [HuggingFace](https://huggingface.co/datasets/datamol-io/safe-gpt), <br>  
**Data Collection Method by dataset:** Automated <br>  
**Labeling Method by dataset:** Automated <br>  
**Properties:** 1.1B SAFE strings consist of various molecule types (drug-like compounds, peptides, multi-fragment molecules, polymers, reagents and non-small molecules). <br>  
**Dataset License(s):** [CC-BY-4.0](https://github.com/datamol-io/safe/blob/main/DATA_LICENSE)  <br>

## Evaluation Dataset:

**Link:** SAFE-DRUGS [GitHub](https://github.com/datamol-io/safe), [HuggingFace](https://huggingface.co/datasets/datamol-io/safe-drugs) <br>  
**Data Collection Method by dataset:** Not Applicable <br>  
**Labeling Method by dataset:** Not Applicable <br>  
**Properties:** SAFE-DRUGS consists of 26 known therapeutic drugs. <br>  
**Dataset License(s):** [CC-BY-4.0](https://github.com/datamol-io/safe/blob/main/DATA_LICENSE) <br>

## Inference:  
**Engine:** PyTorch <br>  
**Test Hardware:** A6000, A100, L40, L40S, H100<br>

# Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards [Insert Link to Model Card++ subcards here].

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Subcards

## Bias

|Field|Response|  
|:---|:---|  
|Participation considerations from adversely impacted groups ([protected classes](https://www.senate.ca.gov/content/protected-classes)) in model design and testing|Not Applicable|  
|Measures taken to mitigate against unwanted bias|Not Applicable|

## Explainability

|Field|Response|  
|:---|:---|  
|Intended Application(s) & Domain(s)|Molecular drug discovery and design|  
|Model Type|Molecular sequence generation|  
|Intended Users|Developers in the academic or pharmaceutical industries who build artificial intelligence applications to perform property guided molecule optimization and novel molecule generation.|  
|Output|Text (molecule sequence)|  
|Describe how the model works|From the input of a molecular sequence (SAFE format) with masks (masking tokens), the neural network model (Transformer & BERT architecture) will one-by-one predict the best characters (valid tokens in the chemistry vocabulary) to replace the masking tokens, until all masks are replaced, which is an unmasking process by discrete diffusion.||  
|Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of|Not Applicable|  
|Technical Limitations| Model may not perform well on sequences that are highly divergent from the ZINC-15 dataset.|  
|Verified to have met prescribed quality standards?|Yes|  
|Performance Metrics|Validity, Uniqueness, Diversity, Central Distance, Qualified Ratio.|  
|Potential Known Risks|The model may produce molecules that are difficult or impossible in synthesis.|  
|Licensing & Terms of Use|The NIM container is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/); and the use of this model is governed by the [AI Foundation Models Community License Agreement](https://docs.nvidia.com/ai-foundation-models-community-license.pdf).|

## Privacy

|Field|Response|  
|:---|:---|  
|Generatable or reverse engineerable personal data?|No|  
|Personal data used to create this model?|No|  
|How often is dataset reviewed?|Before Every Release|  
|Is there provenance for all datasets used in training?|Yes|  
|Does data labeling (annotation, metadata) comply with privacy laws?|Yes|  
|Applicable Privacy Policy|[NVIDIA Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/)|

## Safety

|Field|Response|  
|:---|:---|  
|Model Application(s)|Molecular drug discovery and design|  
|Describe life critical application (if present)|Experimental drug discovery and medicine. Should not be used for life-critical use cases per [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/).|  
|Use Case Restrictions|Abide by [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/); the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/); and the [AI Foundation Models Community License Agreement](https://docs.nvidia.com/ai-foundation-models-community-license.pdf).|  
|Model and Dataset Restrictions|The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to.|