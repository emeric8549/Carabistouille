# Hackathon

In this project, I present the work I have done with a colleague during a hackathon in June 2021.  
This hackathon has been created by Veolia and the foundation Tara Océan with the aim of classifying plankton images among more than 80 classes. One of the main challenges was to build a frugal model that was able to run on a Raspberry Pi. Indeed, the winners saw their project being used by a team of researchers on a boat in the Atlantic Ocean, where it was used to track the evolution of the plankton population at various locations.  

After a whole weekend of coding using multiple approaches, our team won the hackathon and we have been able to meet the team of Tara Océan to learn more about their jobs and mission.  

### Methodology

To achieve good results on a 80+ classes datasets, we knew that we had to use SOTA models. However, the main problem of those models is that they usually use a huge amount of parameters which was not suitable with our task.  
Reading the literature, we learnt about the knowledge distillation technique. This method works the same way as a teacher transfering their knowledge to a student. The idea is to train a first big model able to capture a lot of information in the data. Once we achieve good performance with this 'teacher', we can train a second model, the 'student', that has to predict both the ground-truth labels along with the distribution of the teacher's prediction. This can be done by mixing two losses function, the first one being the cross-entrppy and the second one the Kullbach-Leibler divergence. Depending on what we would like to improve the most, we introduce a hyperparameter alpha that balance the weight of the two results. 

### Work to do:
- Compare results of student alone and with knowledge distillation
- Get number of parameters for each model