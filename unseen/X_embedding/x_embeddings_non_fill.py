# %%
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
df = pd.read_csv("../data/dataset_combine_unseen.csv",keep_default_na=False, na_values=[''])
df

# %%
# Specify the column range to encode (3rd to 30nd column; Python index 2 to 31)
feature_columns = df.columns[1:30]
feature_columns

# %%
def row_to_structured_text(row):
    return f"""
# Nanomaterial Properties
- Core: {row['Core']}, categorized as a {row['Core type']} nanomaterial.
- Surface Modification: {row['Surface modification']} ({row['Modification type']}).
- Shape and Size: {row['Shape']}, with a primary size of {row['Primary size']} nm (TEM), hydrodynamic size in water of {row['DLS size in water (nm)']} nm, and PdI {row['PdI in water']}.
- Surface Charge in Water: Zeta potential is {row['Zeta potential in water (mV)']} mV.
- In Dispersion Medium: DLS size is {row['DLS size in dispersion medium (nm)']} nm, zeta potential is {row['Zeta potential in dispersion medium (mV)']} mV, PdI is {row['PdI in dispersion medium']}, and the medium is {row['Dispersion medium']}.
- NM Concentration: {row['NM concentration']}.

# Incubation Conditions
- Protein Source: {row['Incubation protein source']} from {row['Protein source organism']}, concentration: {row['Protein source concentration']}.
- Culture Medium: {row['Incubation culture']}.
- Time and Temperature: {row['Incubation time (h)']} hours at {row['Incubation temperature (℃)']} °C.
- Flow Conditions: {row['Incubation flow condition']} flow, with speed {row['Incubation flow speed']}.
- Setting: {row['Incubation setting']} environment.

# Separation Parameters
- Separation Method: {row['Separation method']}.
- If centrifugation is used:
- Centrifugation Speed: {row['Centrifugation speed']} for {row['Centrifugation time (min)']} minutes at {row['Centrifugation temperature (℃)']} °C.
- Repetitions: {row['Centrifugation repetitions']} cycle(s).

# Proteomic Setting
- Proteomic depth: {row['Proteomic depth']} proteins.
""".strip()
texts = df.apply(row_to_structured_text, axis=1).tolist()
texts


# %%
def get_detailed_instruct(task_description: str, query: str, context: str) -> str:
    return f'''[Instruction]
{task_description}

[Context]
{context}

[Query]
Below is a structured description of an experimental setting related to nanomaterial exposure in a biological environment. The goal is to encode this input to represent the expected influence on nanomaterial–protein corona affinity.

{query}
'''

# %%
task = '''Given a structured description of an experimental setup involving nanomaterials, your task is to generate an embedding that captures how combinations and interactions among nanomaterial properties, dispersion media, incubation conditions, separation protocols and proteomic setting are expected to influence the resulting protein corona composition and affinity. Pay particular attention to variables that directly or indirectly affect protein adsorption dynamics. Note that "Unknown" indicates missing or unavailable value.'''
context = '''##Protein Corona Background##
When nanomaterials (NMs) enter biological systems, they interact with biomolecules, particularly proteins, forming a "protein corona" on their surface. This corona significantly alters the physical and chemical properties of the nanomaterials, influencing their biological interactions, cellular uptake, toxicity, biodistribution, and overall functionality. The protein composition of the protein corona is affected by many parameters, such as Nanomaterial Properties, Incubation Conditions, Separation Parameters, and Proteomic Setting.

##Parameter Descriptions##
#Nanomaterial Properties#
1. Core: The main composition of the nanomaterial (e.g., gold, silica, etc.).
2. Core type: The specific type or chemical structure of the core (classify the core into seven classes: metal-based, metal oxide-based, polymer-based, lipid-based, carbon-based, core-shell, and other).
3. Surface modification: Modifications applied to the surface of the nanomaterial (e.g., functional chemical groups).
4. Modification type: The specific type of surface modification or functionalization (classify the surface modification into three classes: neutral, cationic, and anionic).
5. Shape: The geometric morphology of the nanomaterial (e.g., spherical, rod-like, etc.).
6. Primary size: The dry size of the nanomaterial core measured using Transmission Electron Microscopy (TEM) or Scanning Electron Microscope (SEM), often representing the dry size. 
7. DLS size in water (nm): The hydrodynamic size of the nanomaterial in water, determined using Dynamic Light Scattering (DLS).
8. Zeta potential in water (mV): The surface charge of the nanomaterial in water, indicating stability and interactions.
9. PdI in water: The polydispersity index in water, reflecting the size distribution uniformity (lower values indicate more uniform particles).
10. DLS size in dispersion medium (nm): The hydrodynamic size of the nanomaterial in a specific dispersion medium.
11. Zeta potential in dispersion medium (mV): The surface charge in the dispersion medium, reflecting particle stability under those conditions.
12. PdI in dispersion medium: The polydispersity index in the dispersion medium, indicating size uniformity.
13. Dispersion medium: The liquid medium in which the nanomaterials are dispersed (e.g., water, buffer, serum). 
14. NM concentration: The concentration of the nanomaterial in the dispersion medium. 

#Incubation Conditions#
15. Incubation protein source: The origin of the protein used for incubation (e.g., human plasma, human serum, fetal bovine serum, or mouse plasma).
16. Protein source organism: The organism from which the protein is derived (e.g., human, bovine, mouse).
17. Protein source concentration: The concentration of plasma or serum in the incubation medium.
18. Incubation culture: Specific culture conditions during incubation (e.g., water, buffer, or DMEM).
19. Incubation time (h): The duration of the incubation period.
20. Incubation temperature (℃): The temperature at which the incubation occurs, typically reflecting physiological or experimental conditions.
21. Incubation flow condition: Whether the incubation is static or under flow conditions (static or flow).
22. Incubation flow speed: The speed of flow during incubation, relevant for dynamic conditions. 
23. Incubation setting: Specifies whether the incubation occurs in in vivo (within a living organism) or in vitro (in a controlled laboratory environment) culture conditions.

#Separation Parameters#
24. Separation method: The method used to separate protein corona (e.g., centrifugation or magnetic separation).
25. Centrifugation speed: The relative centrifugal force applied during centrifugation, expressed in multiples of gravity.
26. Centrifugation time (min): The duration of the centrifugation step.
27. Centrifugation temperature (℃): The temperature during centrifugation, affecting nanomaterial stability and protein binding.
28. Centrifugation repetitions: The number of centrifugation cycles applied to separate bound and unbound proteins.

#Proteomic Setting#
29. Proteomic depth: The number range of proteins identified in a sample, reflecting the analytical coverage and resolution of the protein corona composition. A greater proteomic depth generally corresponds to a lower average relative abundance per protein, as the total signal is distributed across a larger number of identified proteins.
'''

texts_with_instruct = [get_detailed_instruct(task, t, context) for t in texts]
print(texts_with_instruct[0])


# %%
# Load the model locally
model_path = "../../../pretrained_model/Linq-Embed-Mistral"
model = SentenceTransformer(model_path)

# %%
# texts_with_instruct 是一个字符串列表，每个元素是一个结构化 prompt
embeddings = model.encode(texts_with_instruct, batch_size=16, show_progress_bar=True, convert_to_numpy=True)

# %%
np.save("x_embeddings_non_fill.npy", embeddings)

# %%
