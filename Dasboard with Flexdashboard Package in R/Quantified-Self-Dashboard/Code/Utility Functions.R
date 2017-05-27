require(tidyverse)
require(ggthemes)

plotDays = function() {
  days %>% 
    select(day:poms_tot_m) %>% 
    gather(type, pomodoros, -day, -day_of_week) %>% 
    mutate(
      type = substring(type, nchar(type)),
      type = factor(recode(
        type,
        "t" = "Total",
        "w" = "Work",
        "g" = "Growth",
        "h" = "Health",
        "m" = "Misc"
      ), levels = c("Total", "Work", "Growth", "Health", "Misc")),
      day = as.Date(as.character(day), "%Y%m%d")
    ) %>% 
    ggplot(., aes(x = day, y = pomodoros, group = type, color = type)) +
      geom_point() + geom_line() + geom_smooth() + 
      theme_fivethirtyeight() + scale_color_wsj() +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
}