Project 1 Summary 
=================

Overall the class did an excellent job! 

21/29 were 19 or higher!

* 19+ (21 projects): 20, 20, 20, 20, 20, 20, 19.75, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.25, 19.25, 19, 19 
* 18+ (5 projects): 18.75, 18.75, 18.75, 18.5, 18


General Comments: Kudos and Cautions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The scores were very good, and, generally speaking, people are understanding the material, 
we want to mention a few issues that came up. 

Kudos
^^^^^^
1. Several people used ordinal encoding on the columns instead of one-hot encoding. One person even experimented with both 
   ordinal and one-hot encoding to see which one performed best. Excellent job thinking about the data and what might make 
   the most sense for 

2. Several people experimented with other techniques, such as other kinds of ML models. This was not required but we are 
   always happy to see students eager to experiment with new kinds of techniques. A word of caution though: be sure you 
   understand the API you are using, and feel free to talk to us if you're interested in exploring some new API or 
   techniques. 

Cautions 
^^^^^^^^
1. When using AI, make sure you understand what the generated code does. It's clear several people are using 
   ChatGPT, etc., but some of you seemed to 
   Also, please say **please state exactly how you used it**. We did not count off 
   on Project 1 for this but we will be in future projects.  

2. Many people did not realize that they needed to treat columns with invalid values 
   ("?" and "*") -- you can use the ``unique()`` function on a dataframe to see all values.

3. Please consider splitting code into multiple cells and adding comments to the code. This will improve the 
   code readability. 
