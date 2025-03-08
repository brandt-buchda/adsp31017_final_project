setwd('~/Downloads/University of Chicago/Machine Learning 1/')
require(tidyverse)
require(data.table)


### Rotten Tomatoes

reviews <- fread("Rotten/rotten_tomatoes_movie_reviews.csv")
movies <- fread("Rotten/rotten_tomatoes_movies.csv")

### Meta 

## Cast / Crew
credits <- fread("Meta_Movie/credits.csv")

## key words
keywords <- fread("Meta_Movie/keywords.csv")

## link small
link_small <- fread("Meta_Movie/links_small.csv")

## link big
links <- fread("Meta_Movie/links.csv")

## Metadata
meta <- read_csv("Meta_Movie/movies_metadata.csv")


### Wiki 

wiki_plot <- fread("wiki_movie_plots_deduped.csv")
names(wiki_plot)

movies$`Release Year` <- year(movies$releaseDateTheaters)
### Merge with Rotten Tomatoes Meta ###
names(wiki_plot)[2] <- 'title'
movie_wiki <- merge(movies, wiki_plot, by = c('title', 'Release Year'), all.y = TRUE)

movie_wiki <- movie_wiki %>%
  filter(
    !is.na(releaseDateTheaters)
  )

### Merge with Credits

meta_credit <-merge(meta,credits ,by = c('id'))


meta_credit$`Release Year` <- year(meta_credit$release_date)

### Final Data Clean / EDA
tot <- merge(movie_wiki, meta_credit, by = c("title", 'Release Year'), all.x = TRUE)

tot <- tot %>% 
  select(
    -c(id.x, id.y,poster_path, homepage,
       status, video, imdb_id, `Wiki Page`, adult)
  )

tot1 <- tot %>% 
  select(
    -c(audienceScore, tomatoMeter, popularity, 
       vote_average, vote_count)
  )

tot1 <- tot1 %>% select(sort(names(.)))


### 17 Again example, for cast ###
tot1 %>% 
  filter(
    title == "17 Again"
  ) %>% 
  select(cast, Cast, crew)
###


tot2 <- tot1 %>% 
  select(
    -c(cast, Director, Genre, genres, original_language, 
       original_title, releaseDateStreaming)
  )

tot3 <- tot2 %>% 
  filter(boxOffice != "", budget != 0, !is.na(budget))


tot4 <- tot3 %>% 
  select(
    -c(revenue, release_date, runtime)
  )

tot4 <- tot4 %>% 
  select(title, boxOffice, everything())

tot5_distinct <- tot4 %>%
  distinct(title, .keep_all = TRUE)


### Visualize 

unique(substr(tot5_distinct$boxOffice, nchar(tot5_distinct$boxOffice),
              nchar(tot5_distinct$boxOffice)))

tot5_distinct <- tot5_distinct %>%
  mutate(boxOfficeNumeric = case_when(
    str_detect(boxOffice, "M") ~ as.numeric(str_remove_all(boxOffice, "[$M]")) * 1e6,
    str_detect(boxOffice, "K") ~ as.numeric(str_remove_all(boxOffice, "[$K]")) * 1e3,
    TRUE ~ NA_real_
  ))

require(scales)

tot5_distinct %>%
  ggplot(aes(boxOfficeNumeric)) + 
  geom_histogram(bins = 50, color = 'darkblue', fill = 'skyblue') +
  scale_x_continuous(
    labels = comma) + 
  labs(
    x = "Box Office Revenue", 
    y = "Frequency"
  ) + 
  theme_minimal()



tot5_distinct %>%
  ggplot(aes(budget,boxOfficeNumeric)) + 
  geom_point() +
  scale_x_continuous(
    labels = comma) + 
  scale_y_continuous(
    labels = comma) + 
  labs(
    x = "Movie Budget", 
    y = "Box Office Revenue"
  ) + 
  theme_minimal()


### A part of collection series / known brand. 
tot5_distinct %>%
  summarise(
    Is_Collection = paste0(round(sum(!is.na(belongs_to_collection)) / nrow(tot5_distinct),2), '%'),
    Not_Collection = paste0(round(sum(is.na(belongs_to_collection)) / nrow(tot5_distinct), 2),  '%')
  )


### part of collection
tot5_distinct %>% 
  mutate(
    Is_Collection = ifelse(!is.na(belongs_to_collection), 1, 0)
  ) %>% 
  ggplot(aes(x = as.factor(Is_Collection), y = boxOfficeNumeric)) + 
  geom_boxplot() + 
  labs(
    x = "Part of Collection", 
    y = "Box Office Revenue"
  ) + 
  scale_y_continuous(labels = comma) + 
  theme_minimal()



tot5_distinct %>%
  ggplot(aes(budget)) + 
  geom_histogram(bins = 50, color = 'darkblue', fill = 'skyblue') +
  scale_x_continuous(
    labels = comma) + 
  labs(
    x = "Movie Budget", 
    y = "Frequency"
  ) + 
  theme_minimal()



count_distinct_names <- function(column) {
  column %>% 
    str_split(",\\s*") %>%  # split each string at commas (ignoring spaces)
    unlist() %>%            # flatten the list into a vector
    str_trim() %>%          # trim any extra whitespace  # remove any empty strings
    unique() %>%            # get unique names
    length()                # count them
}

# Calculate distinct counts for each column
cast_count <- tot5_distinct %>% pull(Cast) %>% count_distinct_names()
director_count <- tot5_distinct %>% pull(director) %>% count_distinct_names()
distributor_count <- tot5_distinct %>% pull(distributor) %>% count_distinct_names()

# Create a table with the results
result_table <- tibble(
  Role = c("Cast", "Director", "Distributor"),
  `Unique Members` = c(cast_count, director_count, distributor_count)
)


count_distinct_names(tot5_distinct$genre)



genre_avg <- tot5_distinct %>%
  # Remove extraneous quotes
  mutate(genre = str_remove_all(genre, '"')) %>%
  # Split genres into multiple rows
  separate_rows(genre, sep = ",\\s*") %>%
  # Trim any extra whitespace
  mutate(genre = str_trim(genre)) %>%
  # Group by the cleaned genre and calculate the mean box office revenue
  group_by(genre) %>%
  summarize(avg_boxOffice = mean(boxOfficeNumeric, na.rm = TRUE)) %>%
  ungroup()

genre_avg %>%
  mutate(genre = fct_reorder(genre, avg_boxOffice, .desc = TRUE)) %>%
  ggplot(aes(x = genre, y = avg_boxOffice)) +
  geom_col(fill = 'skyblue', color = 'darkblue') +
  coord_flip() +  # flips the axes for better readability
  labs(
    x = "Genre",
    y = "Average Box Office Revenue"
  ) +
  theme_minimal() + 
  scale_y_continuous(labels = comma)



tot5_distinct %>%
  ggplot(aes(x = runtimeMinutes / 60)) + 
  geom_histogram(bins = 50, fill = 'skyblue', color = 'darkblue') +
  labs(
    x = "Movie Run Time (hours)", 
    y = "Frequency"
  ) +
  theme_minimal() + 
  scale_x_continuous(breaks = seq(1, 4, .25))



tot5_distinct %>%
  ggplot(aes(runtimeMinutes,boxOfficeNumeric)) + 
  geom_point() +
  scale_x_continuous(
    labels = comma) + 
  scale_y_continuous(
    labels = comma) + 
  labs(
    x = "Movie Runtime (minutes)", 
    y = "Box Office Revenue"
  ) + 
  theme_minimal()


###

## lot of non ratings
table(tot5_distinct$rating)


## crew members
head(tot5_distinct$crew, 1)



tot5_distinct <- tot5_distinct %>%
  mutate(unique_crew_count = map_int(crew, function(x) {
    # Extract all occurrences of crew member names from the string using regex
    matches <- str_match_all(x, "'name':\\s*'([^']+)'")[[1]]
    
    # If no matches are found, return 0, otherwise count unique names
    if(nrow(matches) == 0) {
      0
    } else {
      length(unique(matches[,2]))
    }
  }))


tot5_distinct %>% 
  ggplot(aes(unique_crew_count)) + 
  geom_histogram(bins = 50, fill = 'skyblue', color = 'darkblue') + 
  labs(
    x = "Number of Crew Members", 
    y = "Frequency"
  ) + 
  theme_minimal()




tot5_distinct %>%
  ggplot(aes(unique_crew_count,boxOfficeNumeric)) + 
  geom_point() +
  scale_x_continuous(
    labels = comma) + 
  scale_y_continuous(
    labels = comma) + 
  labs(
    x = "Number of Crew Members", 
    y = "Box Office Revenue"
  ) + 
  theme_minimal()




tot5_distinct <- tot5_distinct %>%
  mutate(language_count = map_int(spoken_languages, function(x) {
    # Extract language names using regex
    matches <- str_match_all(x, "'name':\\s*'([^']+)'")[[1]]
    if(nrow(matches) == 0) {
      0
    } else {
      length(unique(matches[, 2]))
    }
  }))

# Step 2: Calculate the average boxOfficeNumeric for each language_count
avg_boxoffice <- tot5_distinct %>%
  group_by(language_count) %>%
  summarize(avg_boxOffice = mean(boxOfficeNumeric, na.rm = TRUE))

# Step 3: Create a bar plot of average box office numeric by number of languages
ggplot(avg_boxoffice, aes(x = factor(language_count), y = avg_boxOffice)) +
  geom_bar(stat = "identity", fill = 'skyblue', color = 'darkblue') +
  labs(
    x = "Number of Spoken Languages", 
    y = "Average Box Office Numeric"
  ) +
  theme_minimal() + 
  scale_y_continuous(labels = comma)



head(tot5_distinct$Plot,1)


### plot

tot5_distinct %>% 
  mutate(
    Plot_Length = nchar(Plot)
  ) %>% 
  ggplot(
    aes(Plot_Length)
  ) +
  geom_histogram(bins = 50, fill = 'skyblue',
                 color = 'darkblue') + 
  theme_minimal() + 
  scale_x_continuous(breaks = seq(0,20000, 1000)) + 
  labs(
    x = "Plot Characters", 
    y = "Frequency"
  )



head(tot5_distinct$Cast, 1)
