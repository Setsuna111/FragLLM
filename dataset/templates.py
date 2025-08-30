# referring description
Frag_Act_Des = [
    "Given the following protein sequence {full_sequence}, could you provide a detailed description of the active site located in the region {fragment}? Specifically, I am interested in its biochemical function, catalytic mechanism, key functional residues, any cofactor requirements, and any family-specific structural or functional context, as well as any other relevant characteristics if available."
]

Frag_BindI_Des = [
    "Given the following protein sequence {full_sequence}, could you provide a detailed description of the binding site located in the region {fragment}? Specifically, I am interested in its binding mode, ligand specificity, key functional residues, any cofactor requirements, family-specific structural or functional context, conservation or distribution pattern, and any other relevant characteristics if available."
]

Frag_Dom_Des = [
    "Given the following protein sequence {full_sequence}, could you provide a detailed description of the domain located in the region {fragment}? Specifically, I am interested in its functional role, domain classification/family context, any associated Gene Ontology or Enzyme Commission annotation, representative distribution across species or protein examples, and any additional relevant characteristics if available."
]   

Frag_Evo_Des = [
    "Given the following protein sequence {full_sequence}, could you provide a detailed description of the evolutionary conserved site located in the region {fragment}? Specifically, I am interested in its conserved residue context, potential functional or structural role, family- or subfamily-specific occurrence, general location within the protein or domain framework, and any other relevant characteristics if available."
]

Frag_Motif_Des = [
    "Given the following protein sequence {full_sequence}, could you provide a detailed description of the motif domain located in the region {fragment}? Specifically, I am interested in its functional role, domain classification/family context, any associated Gene Ontology or Enzyme Commission annotation, representative distribution across species or protein examples, and any additional relevant characteristics if available."
]

# referring classification
Frag_Class = [
    "Given the following protein sequence {full_sequence}, could you provide the category name of the {task_name} located in the region {fragment}?"
]

Class_Answer = [
    "It is the {class_name}."
]

# Grounding Single Class
Frag_Ground_Single = [
    "The protein sequence is {full_sequence}, with a total length of {N}, and the amino acid sequence index starts from 0. Please provide the start and end positions of the {class_name} in this sequence."
]

Grounding_Answer_Single = [
    " The {class_name} is located at {position}."
]

# Grounding Full Protein
Frag_Ground_Group = [
    "The protein sequence is {full_sequence}, with a total length of {N}, and the amino acid sequence index starts from 0. Identify all {task_name} categories contained in the protein and provide their start and end positions in the sequence."
]

Grounding_Answer_Group = [ 
    "Alright, I've analyzed the protein sequence you provided. Here are the {task_name} categories I've identified and their positions: {contents}." # Class_1: Position1, Position2, and Position3; Class_2: Position4, Position5, and Position6; ...
    ]

# Function
ProteinFunction = [
    "Protein name: {fullname}; Taxon: {taxon}; Sequence embeddings: {full_sequence}. Please describe its function clearly and concisely in professional language."
]


