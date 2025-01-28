# Final_Project_Python
Final Project Repository for Python Programming

Overview (From Proposal):

This is an idea around building a tool that can run automated value creation analysis on a mid-sized software business. My old firm wants me to do this for them so I’m excited at the idea of it and would actually have a user for it
This tool would take in 15-20 structured input files, along with an income statement for the business, and do three things
Use the files as input to 15-20 automated analyses that identify if there are opportunities to improve growth or reduce costs at the business, by evaluating specific operational / financial data and comparing that performance to what we know best in class looks like for a software business at that scale
Integrate all 15 analyses into a past-and-forward looking income statement that shows, both, how much more profitable a business could have been historically, and how much better it could perform if it executed these various value creation analyses
Take this growth in EBITDA and growth rate and analyze the enterprise value impacts of these analyses
The web app would have the following functionality:
User can download any or all blank file templates
User can upload as many completed files as they want
App has a pre-set list of benchmarks it compares values against
User can select which analyses they want to run, and what implementation timeline they want
User receives individual analyses files, and integrated income statement


Week 4:
-Spend a few hours exploring PANDAS and Openpyxl excel libraries to understand how they represent data sheets, how to iteratre throughan excel, how to write and read excels, and generally understand how to use them

Week 5:
Decide on data structures for input files (probably 5-10), income statement, and the intermediate steps of "improved" income statement and value creation bridge. Also get a mini-web app that lets users download file tempaltes and re-uplaod them when filled out with data

Week 6:
Write analysis logic for each type of file. Also impose basic error handling for files - e.g. what happens if someone puts a comma in a cell with a number. Begin toying with output page / visuals

Week 7: 
Integarte analysis into before and after income statements, with ability to downlaod intermediate files and final file

Week 8: 
Begin building in the UI/UX elements of the app, and make it a smoother user experience

Week 9:
If extra time, begin exploring taking in unstructured files and sseeing if we can easily get them into a standard excel format using an OpenAI API or something else. Also explore having OpenAI give "summareis and action plans" for each step







