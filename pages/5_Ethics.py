import streamlit as st
import webbrowser

st.title("Ethics")
st.subheader("Ethical checklist")

# General Project Information
st.subheader("General Project Information")
st.write("What is the project about and what is its goal? ✅")
st.write("What kind of data will you be using? ✅")
st.write("Who might be affected by the project? ✅")
st.write("What are the benefits of the project? ✅")
st.write("What are the problems or concerns that might arise in connection with this project? ✅")

# Fundamental Rights
st.subheader("Fundamental Rights")
st.write("Does the AI system potentially negatively discriminate against people? ✅")
st.write("Which laws and regulations apply to your project? ✅")

# Human Agency and Oversight
st.subheader("Human Agency and Oversight")
st.write("Could the AI system affect human autonomy? ✅")
st.write("Is the AI system designed to interact, guide, or take decisions by human end-users that affect humans or society? ✅")
st.write("Could the AI system generate confusion for some or all end-users or subjects on whether they are interacting with a human or AI system? ✅")
st.write("Does the AI system risk creating human attachment, stimulating addictive behavior, or manipulating user behavior? ✅")
st.write("Did you take any specific oversight and control measures to reflect the self-learning or autonomous nature of the AI system? ✅")
st.write("Did you establish any detection and response mechanisms for undesirable adverse effects of the AI system for the end-user or subject? ✅")
st.write("Will the results of the analysis be evaluated by a human before being implemented? ✅")

# Technical Robustness and Safety
st.subheader("Technical Robustness and Safety")
st.write("Is the AI system certified for cybersecurity or is it compliant with specific security standards? ✅")
st.write("How exposed is the AI system to cyber-attacks? ✅")
st.write("Did you put measures in place to ensure the integrity, robustness, and overall security of the AI system against potential attacks over its lifecycle? ✅")
st.write("Was risk assessment done for such as design or technical faults, defects, outages, attacks, misuse, inappropriate or malicious use? ✅")
st.write("Did you define risks, risk metrics, and risk levels of the AI system in each specific use case? ✅")
st.write("Did you identify the possible threats to the AI system and the possible consequences? ✅")
st.write("Did you assess the dependency of a critical AI system’s decisions on its stable and reliable behavior? ✅")
st.write("Did you plan fault tolerance? ✅")
st.write("Could a low level of accuracy of the AI system result in critical, adversarial, or damaging consequences? ✅")
st.write("In what ways have you checked the quality of the data? ✅")
st.write("Did you put in place a series of steps to monitor and document the AI system’s accuracy? ✅")
st.write("Did you put processes in place to ensure that the level of accuracy of the AI system to be expected by end-users and/or subjects is properly communicated? ✅")
st.write("Could the AI system cause critical, adversarial, or damaging consequences? ✅")
st.write("Did you consider whether the AI system's operation can invalidate the data? ✅")
st.write("Did you put in place verification and validation methods and documentation to evaluate and ensure different aspects of the AI system’s reliability and reproducibility? ✅")

# Privacy and Data Governance
st.subheader("Privacy and Data Governance")
st.write("Is your AI system being trained, or was it developed, by using or processing personal data (including special categories of personal data)? ✅")
st.write("Did you consider the impact of the AI system on the right to privacy? ✅")
st.write("Did you consider the privacy and data protection implications of the AI system's non-personal training-data or other processed non-personal data? ✅")
st.write("Did you align the AI system with relevant standards or widely adopted protocols for (daily) data management and governance? ✅")


# Transparency
st.subheader("Transparency")
st.write("Did you establish mechanisms to inform users about the purpose, criteria, and limitations of the decision(s) generated by the AI system? ✅")
st.write("Did you put in place measures that address the traceability of the AI system during its entire lifecycle? ✅")
st.write("Did you explain the decision(s) of the AI system to the users? ✅")
st.write("What is the communication strategy with regard to this project? ✅")
st.write("Are you gathering the information that is appropriate for the purposes of your project? ✅")

# Diversity, Non-discrimination, and Fairness
st.subheader("Diversity, Non-discrimination, and Fairness")
st.write("Did you establish a strategy or a set of procedures to avoid creating or reinforcing unfair bias in the AI system? ✅")
st.write("Is there a risk that the project will unintentionally create incentives for undesirable behavior? ✅")
st.write("Did you consider diversity and representativeness of end-users and/or subjects in the data? ✅")
st.write("Is your definition of fairness commonly used and implemented in any phase of the process of setting up the AI system? ✅")
st.write("Did you ensure a mechanism that allows for the flagging of issues related to bias, discrimination, or poor performance of the AI system? ✅")
st.write("Did you ensure that Universal Design principles are taken into account during every step of the planning and development process, if applicable? ✅")
st.write("Did you consider a mechanism to include the participation of the widest range of possible stakeholders in the AI system’s design and development? ✅")

# Societal and Environmental Well-being
st.subheader("Societal and Environmental Well-being")
st.write("Could the AI system create the risk of de-skilling of the workforce? ✅")
st.write("Does the system promote or require new (digital) skills? ✅")
st.write("Does the AI system impact human work and work arrangements? ✅")
st.write("Could the AI system have a negative impact on society at large or democracy? ✅")

# Accountability
st.subheader("Accountability")
st.write("Did you establish mechanisms that facilitate the AI system’s auditability? ✅")
st.write("Did you ensure that the AI system can be audited by independent third parties? ✅")

report_link = "https://github.com/JakubCyba223860/streamlit-codespaces/blob/main/Ethical_report.pdf"

image = Image.open('pages/MicrosoftTeams-image (1).png')
st.image(image, caption='AI Canvas', use_column_width=True)

if st.button("Report Link"):
    webbrowser.open(report_link)
    
st.title("Feedback Form")

feedback = st.text_area("Feedback", height=150)

if st.button("Submit"):
    # Validate form field
    if not feedback:
        st.error("Please enter your feedback.")
    else:
        # Save feedback to a file
        with open("feedback.txt", "a") as file:
            file.write(f"Feedback: {feedback}\n\n")
        # Show confirmation message
        st.success("Thank you for your feedback!")