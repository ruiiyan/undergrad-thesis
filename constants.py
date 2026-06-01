import re

# File paths
DATASET_PATH = 'data/ENGG_2407_STAR_REFLECTIONS.xlsx'

# Column mapping
COL_MAP = {
    0: 'unit',
    1: 'submission_ref',
    2: 'student_id',
    3: 'degree',
    4: 'gender',
    5: 'status',
    6: 'start_time',
    7: 'end_time',
    8: 'time_taken',
    9: 'grade',
    10: 'topic',
    11: 'star_text',
    12: 'pfr_l_text'
}

# Based on Rubric: 20S2_E2K_3K_Reflective writing_v2.docx
GRADE_DISTRIBUTION = {
    "fail": (0, 4.9),
    "pass": (5.0, 6.4), 
    "credit": (6.5, 7.4), 
    "distinction": (7.5, 8.4), 
    "high_distinction": (8.5, 10.0)
}

STAR_HEADERS = {
    'situation': re.compile(
        r'(?<!\w)SITUA?TION\s*(?:\([AB]\))?\s*[:/]',
        re.IGNORECASE
    ),
    'task_action': re.compile(
        r'TASK\s*[/&]?\s*ACTIONS?\s*(?:\([AB]\))?\s*[:/]'
        r'|TASKS?\s*/?\s*ACTIONS?\s*[:/]'
        r'|ACTIONS?\s*/?\s*TASKS?\s*[:/]'
        r'|TASK\s+OR\s+ACTION\s*[:/]'
        r'|T\s*/\s*A\s*[:/]'
        r'|(?<!\w)TASK\s*[:/](?!\s*[/&]\s*ACTION)',  # TASK: only if not followed by /ACTION
        re.IGNORECASE
    ),
    'result': re.compile(
        r'(?<!\w)RESULTS?\s*(?:\([AB]\))?\s*[:/]',
        re.IGNORECASE
    ),
}

AM_HEADERS = {
    'pfr': re.compile(
        r'PERSONAL\s+FEEL(?:ING|INGS|L(?:ING)?)?\s*[/\s]?\s*REFL?E(?:CT(?:ION|ING|IONS)|XION)S?\s*(?:\([AB]\))?\s*[:/]'
        r'|PERSONAL\s+FEEL(?:ING|INGS)?\s*[:/]'
        r'|PERSONAL\s+REFLECTION\s*[:/]'
        r'|FEEL(?:ING|INGS|S)\s*/?\s*REFLECTION\s*[:/]'
        r'|HOW\s+(?:I|WE)\s+FELT\s*[:/]'
        r'|PERSON\s+FEEL(?:ING|INGS)?\s*/?\s*REFLECTION\s*[:/]'
        r'|ATKINS\s+AND\s+MURPHY\s+MODEL\s+OF\s+REFLECTION\s*[:/]'
        r'|REFLECTION\s+AND\s+PERSONAL\s+FEELING\s*[:/]'
        r'|WHAT\s+WE\s+THINK\s+AND\s+FEEL\s*[:/]'
        r'|(?<!\w)REFLECTION\s*[:/]',
        re.IGNORECASE
    ),
    'learning': re.compile(
        r'(?<!\w)LEARNI?NGS?\s*(?:OUTCOME|EXPERIENCE|FROM\s+THE\s+EXPERIENCE)?\s*(?:\([AB]\))?\s*[:/]'
        r'|WHAT\s+(?:I|WE)\s+(?:HAVE\s+)?LEARN(?:ED|T)\s*[:/]'
        r'|IDENTIFY\s+(?:ANY\s+)?LEARNING\s*[:/]'
        r'|ARTICULATE\s+LEARNING\s*[:/]',
        re.IGNORECASE
    ),
}

# Model settings
SBERT_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'claude-sonnet-4-6'
LLM_TEMPERATURE = 0

# Pipeline settings
TOP_K_NEIGHBOURS = 10  # to be tuned empirically in Chapter 4

SECTIONS = ['situation', 'task_action', 'result', 'pfr', 'learning']
THEMATIC_SECTIONS = ['situation', 'task_action', 'pfr']
COGNITIVE_SECTIONS = ['result', 'learning']

MODELS_TO_COMPARE = [
    'all-MiniLM-L6-v2',        # 384d — baseline
    'all-mpnet-base-v2',        # 768d — general purpose
    'paraphrase-mpnet-base-v2', # 768d — paraphrase optimised
    'multi-qa-MiniLM-L6-cos-v1',
    'all-MiniLM-L12-v2',        # 768d - larger version of L6 MiniLM
]

# UMAP — best from experiments
UMAP_CONFIG = {
    'n_components': 10,
    'n_neighbors': 20,
    'min_dist': 0.0,
    'metric': 'cosine',
    'random_state': 42
}

# HDBSCAN — best from experiments
HDBSCAN_CONFIG = {
    'min_cluster_size': 10,
    'min_samples': 1,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
    'prediction_data': True
}

# Comparative methods
KMEANS_K_RANGE    = range(5, 25, 5)

# Confidence gate matrix: (s1_band, s2_band) -> (confidence, flag_for_review)
CONFIDENCE_GATE = {
    ('high', 'high'): ('high',   False),
    ('mid',  'mid'):  ('high',   False),
    ('low',  'low'):  ('high',   False),
    ('high', 'mid'):  ('medium', False),
    ('mid',  'high'): ('medium', False),
    ('mid',  'low'):  ('medium', True),
    ('low',  'mid'):  ('medium', True),
    ('high', 'low'):  ('low',    True),
    ('low',  'high'): ('low',    True),
}

# ----------------
# Online Phase
# ----------------

ONLINE_PARAMETERS = {
    "k"                    : 15,  #, How many reflections we used to reflect 
    "alpha"                        : 0.1,  # 
    "min_multiplier"               : 0.3,  #
    "s1_weight"                    : 0.3,  # The weighting of Signal 1 of the Final Grade
    "disparity_threshold"          : 2.0,
    "similarity_threshold"         : 0.4,
    "cluster_size_threshold"       : 15
}

S1_WEIGHT = 0.3
S2_WEIGHT = 0.7

S2_RESULT_WEIGHT   = 0.2
S2_PFR_WEIGHT      = 0.4
S2_LEARNING_WEIGHT = 0.4

# Confidence gate band thresholds
S1_LOW_MAX  = 6.5
S1_HIGH_MIN = 7.5
S2_LOW_MAX  = 2      # Bloom level
S2_HIGH_MIN = 5      # Bloom level

BLOOM_TO_GRADE = {
    1: 2.5,    # Fail
    2: 5.7,    # Pass
    3: 7.0,    # Credit
    4: 8.0,    # Distinction
    5: 9.25,   # HD
    6: 10.0,   # theoretical max
}

EXPECTED_BLOOM = {
    'low':  (1.0, 2.0),   # Fail/Pass → expect Bloom 1-2
    'mid':  (2.0, 3.5),   # Credit/Distinction → expect Bloom 2-3.5
    'high': (3.5, 5.0),   # HD → expect Bloom 3.5-5
}

GRADE_TOLERANCE = 1.5

EXAMPLE_REFLECTIONS = [
    {
        "id": 1,
        "grade": 2,
        "topic": "teamwork",
        "situation": "Our engineering group had to finish a small design assignment together. We met a few times during the semester.",
        "task_action": "I mostly waited for other people to tell me what to do. I completed one small section near the deadline.",
        "result": "The assignment was submitted late. The group was not very happy with the final work.",
        "pfr": "I thought the group work was difficult and stressful. I did not really enjoy working with others.",
        "learning": "I learned that teamwork is important and communication matters."
    },
    {
        "id": 2,
        "grade": 3,
        "topic": "time management",
        "situation": "I had several lab reports and quizzes due during the same week. I found it hard to balance everything.",
        "task_action": "I tried to work on tasks whenever I had spare time. I stayed up late before the deadline to finish the report.",
        "result": "I submitted the work but some parts were rushed. My marks were lower than expected.",
        "pfr": "I felt stressed because I left too much work until the end. I was frustrated with how I handled my time.",
        "learning": "I learned I should start assignments earlier and manage my time better."
    },
    {
        "id": 3,
        "grade": 4,
        "topic": "communication",
        "situation": "During a tutorial presentation, I had to explain our circuit design to the class. I was nervous because I do not usually speak in front of people.",
        "task_action": "I prepared some slides quickly and read from them during the presentation. I did not practice much beforehand.",
        "result": "The presentation finished on time but I struggled to answer questions from the tutor. Some classmates seemed confused.",
        "pfr": "I felt uncomfortable speaking in front of others. I realised I was not very confident when explaining technical ideas.",
        "learning": "I learned that preparation and practice are important for communication."
    },
    {
        "id": 4,
        "grade": 5,
        "topic": "problem solving",
        "situation": "In a programming lab, my code for controlling a robotic arm kept producing incorrect movements. This delayed the progress of the task.",
        "task_action": "I checked the code several times and asked another student for help. I eventually found that one of the variables had the wrong value.",
        "result": "The robotic arm started working properly after the fix. However, I lost a lot of time troubleshooting.",
        "pfr": "I felt frustrated at first because I could not immediately identify the issue. After solving it, I felt more confident in debugging.",
        "learning": "I learned that checking smaller parts of the code step by step can help identify problems more efficiently."
    },
    {
        "id": 5,
        "grade": 5.5,
        "topic": "technical drawing",
        "situation": "Our class required us to create a detailed CAD drawing for a mechanical component. I had limited experience using the software.",
        "task_action": "I followed the tutorial examples and spent extra time adjusting the dimensions and views. I also watched online videos to understand some commands.",
        "result": "The final drawing met the basic requirements but still contained minor formatting issues. I received feedback about improving accuracy.",
        "pfr": "I initially felt overwhelmed by the software because there were many tools and settings. Over time I became more comfortable using the program.",
        "learning": "I learned that technical drawing requires patience and attention to detail, especially when working with dimensions."
    },
    {
        "id": 6,
        "grade": 6,
        "topic": "adaptability",
        "situation": "Our laboratory session changed from in-person testing to simulation software because equipment was unavailable. This meant our team had to adjust our approach quickly.",
        "task_action": "I learned how to use the simulation software and helped transfer some of our calculations into the program. I also communicated updates to the group chat.",
        "result": "We completed the experiment and submitted the report on time. The simulation results were close to the expected values.",
        "pfr": "I was unsure at first because I preferred working with physical equipment. After using the software more, I became more comfortable adapting to the change.",
        "learning": "I learned that flexibility is important in engineering because unexpected changes can happen during projects."
    },
    {
        "id": 7,
        "grade": 6.3,
        "topic": "professional behaviour",
        "situation": "During an industry networking event, I had to speak with engineers and discuss my project experience. I was concerned about presenting myself professionally.",
        "task_action": "I prepared some questions beforehand and practiced introducing myself. During the event, I made an effort to listen carefully and respond respectfully.",
        "result": "I had several positive conversations and gained useful advice about internships. One engineer also connected with me on LinkedIn.",
        "pfr": "I felt nervous at first but became more confident as the conversations continued. I realised professional behaviour also involves active listening, not only speaking well.",
        "learning": "I learned that preparation and respectful communication help create a more professional impression in engineering environments."
    },
    {
        "id": 8,
        "grade": 6.5,
        "topic": "design process",
        "situation": "Our design project involved creating a water filtration prototype for a first-year engineering subject. The team initially focused on making the design as complex as possible.",
        "task_action": "I suggested simplifying the design after reviewing the marking criteria and testing limitations. I compared different material options and created a basic testing plan for the prototype.",
        "result": "The simplified design was easier to manufacture and performed consistently during testing. Our team completed the project ahead of schedule.",
        "pfr": "I recognised that I originally associated complexity with quality. Through the project, I became more aware that effective engineering design often depends on practicality and reliability.",
        "learning": "I learned that a structured design process helps teams avoid unnecessary complexity and focus on meeting project requirements effectively."
    },
    {
        "id": 9,
        "grade": 6.8,
        "topic": "conflict resolution",
        "situation": "During a group assignment, two team members disagreed about how to divide the workload. The disagreement delayed progress and created tension within the team.",
        "task_action": "I organised a short meeting so everyone could explain their concerns. We then divided tasks based on each person's strengths and agreed on intermediate deadlines.",
        "result": "The group became more organised and communication improved. We completed the assignment with fewer disagreements afterward.",
        "pfr": "I noticed that avoiding conflict at the beginning made the situation worse over time. Addressing the issue directly helped reduce misunderstandings within the team.",
        "learning": "I learned that conflict resolution requires listening carefully to different perspectives and creating clear expectations for everyone involved."
    },
    {
        "id": 10,
        "grade": 7.0,
        "topic": "leadership",
        "situation": "In a renewable energy project, our team struggled to coordinate tasks because members were working at different times. This caused confusion about responsibilities.",
        "task_action": "I volunteered to coordinate weekly updates and created a shared progress tracker. I also checked in with quieter team members to make sure their ideas were included.",
        "result": "The team became more organised and communication improved significantly. We finished the project with a stronger final presentation than expected.",
        "pfr": "I realised leadership was less about directing people and more about maintaining structure and communication. I became more confident facilitating collaboration rather than trying to control every task.",
        "learning": "I learned that effective leadership in engineering teams depends on coordination, accountability, and ensuring all members can contribute."
    },
    {
        "id": 11,
        "grade": 7.2,
        "topic": "ethics",
        "situation": "While analysing experimental data for a materials engineering report, our results did not fully match the expected outcome. Some group members suggested removing inconsistent data points to improve the report.",
        "task_action": "I argued that the data should remain included and proposed discussing the inconsistencies in the analysis section instead. I reviewed the lab procedure to identify possible sources of error.",
        "result": "The report acknowledged the limitations of the experiment while still presenting valid conclusions. The tutor commented positively on the transparency of the discussion.",
        "pfr": "I initially worried that including inconsistent data would weaken the report. However, I recognised that accurate reporting is more important than presenting perfect results.",
        "learning": "I learned that engineering ethics involves being transparent about uncertainty and limitations rather than only focusing on favourable outcomes."
    },
    {
        "id": 12,
        "grade": 7.4,
        "topic": "communication",
        "situation": "Our capstone team needed to explain a technical design proposal to non-technical stakeholders during a review session. Previous meetings had shown that the audience struggled with technical terminology.",
        "task_action": "I redesigned part of the presentation to include diagrams, simplified explanations, and practical examples. I also practiced answering questions using less technical language.",
        "result": "The stakeholders engaged more actively during the session and asked more meaningful questions. The feedback indicated that the proposal was much easier to understand.",
        "pfr": "I realised I had previously assumed that technical accuracy alone made communication effective. Through this experience, I understood that communication must also consider the audience's background and perspective.",
        "learning": "I learned that strong engineering communication requires translating complex concepts into accessible explanations without losing important meaning."
    },
    {
        "id": 13,
        "grade": 7.5,
        "topic": "teamwork",
        "situation": "During a semester-long robotics project, our team initially divided tasks independently to maximise efficiency. However, integration problems emerged because members were making assumptions about each other's work.",
        "task_action": "I proposed introducing short integration reviews at the end of each week where members demonstrated their progress and discussed dependencies. I also encouraged documenting interface decisions so that changes were visible to everyone.",
        "result": "The number of integration errors decreased significantly over the following weeks. The team completed testing earlier than expected and experienced less last-minute rework.",
        "pfr": "I previously believed teamwork mainly depended on individuals completing their assigned tasks independently. Reflecting on this experience, I recognised that coordination mechanisms are equally important because engineering tasks are often interconnected.",
        "learning": "I learned that successful teamwork is not only about contribution volume but also about maintaining shared understanding between team members throughout a project."
    },
    {
        "id": 14,
        "grade": 7.8,
        "topic": "problem solving",
        "situation": "While developing a sensor calibration system, our readings remained inconsistent despite multiple hardware replacements. The issue persisted across several testing sessions.",
        "task_action": "Instead of continuing to replace components, I reviewed the testing environment and identified that temperature fluctuations were influencing the sensor behaviour. I redesigned the testing procedure to include environmental controls and repeated the experiments.",
        "result": "The measurements became significantly more stable after controlling the testing conditions. The revised process also improved the reliability of our final dataset.",
        "pfr": "I realised I had initially approached the problem too narrowly by assuming the fault existed only within the hardware itself. This reflection showed me that engineering problems often emerge from interactions between systems and their environment.",
        "learning": "I learned that effective problem solving requires questioning underlying assumptions and examining external variables rather than focusing only on the most obvious cause."
    },
    {
        "id": 15,
        "grade": 8.0,
        "topic": "adaptability",
        "situation": "Halfway through our engineering design project, the client changed the project requirements to prioritise sustainability and lower manufacturing cost. This invalidated several earlier design decisions.",
        "task_action": "I facilitated a review session where the team reassessed the design objectives and identified which features no longer aligned with the updated requirements. I also created a comparison matrix to evaluate alternative materials and manufacturing methods.",
        "result": "Although we lost time revising the design, the updated prototype better matched the client's priorities and received positive feedback during the final review.",
        "pfr": "Initially, I viewed changing requirements as a disruption that reduced project efficiency. Reflecting on the experience, I recognised that adaptability is a necessary engineering skill because real-world constraints and stakeholder priorities frequently evolve.",
        "learning": "I learned that adaptable teams respond more effectively when they maintain flexible design processes and regularly reassess assumptions instead of becoming attached to early solutions."
    },
    {
        "id": 16,
        "grade": 8.2,
        "topic": "leadership",
        "situation": "During a multidisciplinary engineering competition, our team consisted of members with very different technical backgrounds and communication styles. Early meetings were inefficient because discussions frequently moved off-topic.",
        "task_action": "I introduced structured meeting agendas with clearly defined outcomes and rotated discussion leadership depending on the technical topic. I also summarised decisions at the end of each meeting to ensure shared understanding.",
        "result": "The meetings became shorter and more productive, and team members participated more evenly across discussions. Our team produced a more integrated final solution because communication between disciplines improved.",
        "pfr": "I had previously assumed leadership meant being the primary decision-maker within the team. Through reflection, I recognised that leadership can instead involve designing processes that enable effective collaboration and reduce communication barriers.",
        "learning": "I learned that leadership effectiveness often depends less on authority and more on creating structures that support coordination, accountability, and inclusive participation."
    },
    {
        "id": 17,
        "grade": 8.4,
        "topic": "professional behaviour",
        "situation": "During an internship placement, I attended meetings with engineers, contractors, and project managers discussing delays on a construction project. I noticed that technical disagreements sometimes became defensive and unproductive.",
        "task_action": "I observed how experienced engineers redirected discussions toward evidence, documentation, and shared project goals rather than personal opinions. I intentionally adopted similar communication behaviours during later meetings and focused on clarifying information before responding.",
        "result": "My contributions during discussions became more constructive, and I was trusted with documenting meeting outcomes for the team. Supervisors also commented positively on my professionalism.",
        "pfr": "I previously viewed professionalism mainly as punctuality and respectful behaviour. Reflecting on the placement, I realised professionalism also involves emotional control, evidence-based communication, and maintaining collaboration under pressure.",
        "learning": "I learned that professional behaviour in engineering environments helps maintain trust and decision quality, especially when projects involve uncertainty or conflicting priorities."
    },
    {
        "id": 18,
        "grade": 8.5,
        "topic": "design process",
        "situation": "Our biomedical engineering team developed a prototype assistive device for elderly users. Early user testing showed that participants struggled to operate features that the team considered intuitive.",
        "task_action": "I reviewed the testing feedback and recognised that our design decisions were based primarily on assumptions made by technically experienced students rather than the needs of actual users. I proposed redesigning the interface using simpler controls and introduced short observational testing cycles after each modification.",
        "result": "The revised prototype became easier for participants to use, and task completion rates improved noticeably during later testing sessions. The team also identified usability issues earlier because feedback was incorporated continuously.",
        "pfr": "Before this project, I assumed good engineering design mainly depended on technical functionality and innovation. Reflecting on the testing process, I realised that usability failures often occur because designers project their own experiences onto users. The repeated testing cycles worked because they exposed hidden assumptions before they became embedded in the final design.",
        "learning": "I learned that effective design processes require continuous interaction with end users and iterative feedback mechanisms. Engineering solutions become more successful when designers deliberately test assumptions instead of relying only on technical reasoning."
    },
    {
        "id": 19,
        "grade": 9.0,
        "topic": "ethics",
        "situation": "During a data analysis project involving environmental monitoring, our team discovered that excluding several outlier measurements would make the final model appear significantly more accurate. Some members argued that the outliers were not representative enough to include.",
        "task_action": "I reviewed the collection process and found that the outliers were linked to extreme weather conditions rather than equipment failure. I recommended retaining the data and reframing the analysis to explain how environmental variability affected model reliability. I also suggested documenting the uncertainty more explicitly in the report.",
        "result": "The final report presented a more balanced interpretation of the data and received positive feedback for its transparency. Although the model accuracy appeared lower, the conclusions were considered more credible and realistic.",
        "pfr": "I initially assumed ethical engineering decisions mainly involved avoiding intentional misconduct. Through reflection, I recognised that ethical issues can also emerge subtly through pressure to simplify results or present cleaner conclusions than the evidence supports. The team produced a stronger outcome once we shifted our focus from defending the model to understanding the system behaviour represented by the data.",
        "learning": "I learned that ethical engineering practice depends on transparency about uncertainty and limitations. Reliable decision-making is strengthened when engineers communicate complexity honestly instead of optimising only for favourable outcomes."
    },
    {
        "id": 20,
        "grade": 9.5,
        "topic": "conflict resolution",
        "situation": "In a large engineering project team, disagreements developed between software-focused and hardware-focused members regarding project priorities. Meetings became increasingly defensive because each subgroup believed the other did not understand their constraints.",
        "task_action": "Rather than trying to immediately settle the disagreement, I organised a workshop where each subgroup explained its workflow, dependencies, and technical limitations using practical examples. I also mapped the interactions between hardware and software milestones to show how delays in one area affected the other.",
        "result": "The tone of discussions changed significantly once team members understood the pressures affecting other parts of the project. Collaboration improved, and later meetings focused more on coordinating dependencies than defending positions.",
        "pfr": "I originally assumed conflict resolution involved finding compromises between competing opinions as quickly as possible. Reflecting on the experience, I realised the conflict persisted because the groups lacked a shared mental model of the project system. The workshop succeeded because it shifted the discussion from personal disagreement to understanding interconnected constraints.",
        "learning": "I learned that many engineering conflicts are driven less by personality differences and more by incomplete visibility between specialised domains. Creating mechanisms that improve shared understanding can reduce conflict more effectively than simply negotiating compromises."
    },
    {
        "id": 21,
        "grade": 9.8,
        "topic": "communication",
        "situation": "Our capstone team presented a renewable energy proposal to both academic assessors and industry representatives. During earlier rehearsals, technically detailed explanations caused non-specialist audience members to disengage quickly.",
        "task_action": "I analysed the communication breakdown and recognised that we were presenting information according to how engineers organise knowledge rather than how audiences process unfamiliar concepts. I redesigned sections of the presentation to progressively layer information, beginning with practical impact before introducing technical details. I also incorporated visual comparisons and simplified system diagrams to reduce cognitive overload.",
        "result": "The final presentation generated significantly more engagement and discussion from both technical and non-technical attendees. Industry representatives specifically commented that the proposal was easy to follow without oversimplifying the engineering concepts.",
        "pfr": "Previously, I believed strong technical communication primarily depended on accuracy and completeness. Through reflection, I recognised that communication effectiveness is strongly influenced by cognitive accessibility and audience context. The revised structure worked because it aligned explanations with how listeners gradually build understanding instead of overwhelming them with detail immediately.",
        "learning": "I learned that engineering communication is not simply the transfer of technical information. Effective communication requires deliberately designing explanations that account for audience knowledge, attention, and cognitive processing."
    },
    {
        "id": 22,
        "grade": 10.0,
        "topic": "leadership",
        "situation": "During a year-long engineering research project, our team consistently encountered delays despite all members appearing highly motivated and technically capable. Tasks were completed individually, but integration and decision-making repeatedly stalled progress.",
        "task_action": "I reviewed our workflow and recognised that the problem was not effort but the absence of shared coordination mechanisms. I introduced structured design reviews, dependency tracking, and retrospective discussions after each milestone. Instead of focusing only on immediate technical issues, I encouraged the team to analyse why bottlenecks occurred and how communication patterns influenced delays.",
        "result": "Over time, the team became significantly more proactive in identifying risks and coordinating decisions before problems escalated. Project integration improved, deadlines became more predictable, and the quality of discussions shifted from reactive troubleshooting toward long-term planning.",
        "pfr": "Before this experience, I viewed leadership mainly as motivating individuals and maintaining productivity. Reflecting more deeply, I realised that many project failures emerge from poorly designed systems of collaboration rather than individual performance. The changes were effective because they created feedback loops that allowed the team to continuously detect and adapt to coordination problems instead of repeatedly reacting to symptoms.",
        "learning": "I learned that effective engineering leadership involves shaping the structures and feedback mechanisms through which teams operate. Sustainable improvement occurs when leaders design processes that support reflection, shared awareness, and continuous adaptation rather than relying only on individual effort."
    },
    {
        "id": 23,
        "grade": 5,
        "topic": "teamwork",
        "situation": "My group had to complete a bridge design assignment in class. Some members were more active than others.",
        "task_action": "I completed calculations for one section and shared them with the group. We communicated mostly through messages before the submission.",
        "result": "The project was completed successfully but some tasks were rushed near the deadline.",
        "pfr": "I felt that the teamwork was uneven at times. It was difficult to coordinate everyone's schedules.",
        "learning": "I learned that groups work better when tasks are shared more clearly."
    },
    {
        "id": 24,
        "grade": 6.8,
        "topic": "technical drawing",
        "situation": "In a manufacturing subject, I had to create a complete assembly drawing with accurate tolerances and annotations. Small mistakes in dimensions caused repeated revisions.",
        "task_action": "I reviewed engineering drawing standards more carefully and checked my work against the marking rubric before submission. I also compared my drawings with sample professional drawings provided in class.",
        "result": "The final submission was much more accurate and easier to interpret. My feedback highlighted improved consistency and detail.",
        "pfr": "I realised that I had previously focused more on completing the drawing quickly than ensuring it could be interpreted correctly by others. The revision process showed me how small inaccuracies could affect manufacturing outcomes.",
        "learning": "I learned that technical drawings are communication tools as well as technical documents, so clarity and precision are both essential."
    },
    {
        "id": 25,
        "grade": 8.0,
        "topic": "time management",
        "situation": "During mid-semester, I struggled to balance laboratory preparation, assignment deadlines, and part-time work commitments. My usual approach of prioritising tasks only by deadline caused important preparation work to be delayed.",
        "task_action": "I reviewed how I was allocating time and recognised that I underestimated tasks requiring sustained concentration. I reorganised my schedule by blocking dedicated preparation periods before labs and breaking large assignments into smaller milestones with earlier self-imposed deadlines.",
        "result": "My workload became more manageable, and I completed tasks with less last-minute stress. I also participated more effectively during laboratory sessions because I arrived better prepared.",
        "pfr": "Previously, I assumed time management mainly involved working harder and staying productive for longer periods. Reflecting on this experience, I realised that workload problems often result from poor planning assumptions rather than lack of effort. The revised schedule worked because it accounted for cognitive effort and preparation time rather than only submission dates.",
        "learning": "I learned that effective time management depends on realistically estimating task complexity and creating structured systems that reduce reactive decision-making."
    }
]

