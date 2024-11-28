<h3>combine_json_files.py</h3>

Goes through a folder that contains multiple subfolders, each of which contains multiple json data files.  Combines the json files in each subfolder into 1 big json file.  Replaces the subfolder with the big json file.
<br><br>
run the file with 1 argument passed: the directory which contains subfolders that all have json files in them.

<h3>gather_files.py</h3>

Moves all files from all subdirectories of `source_dir` into `destination_dir`, renaming files to avoid duplicates.
<br><br>
Change source directory and destination directory at the bottom of the file, then run

<h3>get_csv_data.py</h3>

Goes through the 'all the news' dataset csv (or a subsection of it) and finds any entries that contain a stock name which the user provides.  Creates json data files with these entries and outputs them
<br><br>
Change f_name, stock_name, and stock_ticker at the bottom of the file then run it.

<h3>get_news_api_data.py</h3>

Scrapes news article data for the desired stock for the specified date range, then outputs the info as a json data file in the same directory as the file.  Ticker is for creating the json data files.  For the free news api plan you can only go back a month.  the output file will be named <stock name>.json by default, but this can be changed by passing a different name into the function at the bottom.
<br><br>
NOTE: you must add your newsapi API key to line 8 of "data preprocessing helpers/get_news_data/news_api.py"
<br><br>
Edit name, ticker, from_date, to_date, and add a custom output file name if needed at the bottom of the file.  Then run the file.

<h3>make_nested_dirs.py</h3>

Creates a new directory inside `base_dir` for each item in `item_list`.  Inside each of these directories, creates num subdirectories named 1-num.
<br><br>
Change base_directory, names, and num at the bottom of the file according to your needs, then run.


<h3>separate_big_data.py</h3>

Reads a large CSV file in and splits it into chunks of 100k entries each.
<br><br>
Change input_file and output_dir to the path+name of the large csv file and the path for the chunks to be created respectively, then run the file.