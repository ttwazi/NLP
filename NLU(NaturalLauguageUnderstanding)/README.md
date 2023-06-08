# Simplified Q&A System Like Apple Siri
This project implemented a simplified version of Q&A system like Siri or Alexa using NLU(natural language understanding) techniques

## Dataset
Training and Validation dataset used for this project is [2017-06-custom-intent-engines](https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines) from this paper [Coucke A. et al., "Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces.](https://arxiv.org/abs/1805.10190) Please refer to their [github page](https://github.com/sonos/nlu-benchmark/tree/master) for more details. 

## Method
The system is made of two subfunctions, customer intent classification and slot labelling.

1. Customer intent classifier: classify customer intent into one of the six categories: 
 
  > * <h6> SearchCreativeWork (e.g. Find me the I, Robot television show),<br>
  > * <h6> GetWeather (e.g. Is it windy in Boston, MA right now?),
  > * <h6> BookRestaurant (e.g. I want to book a highly rated restaurant for me and my boyfriend tomorrow night),
  > * <h6> AddToPlaylist (e.g. Add Diamonds to my roadtrip playlist)
  > * <h6> RateBook (e.g. Give 6 stars to Of Mice and Men)
  > * <h6> SearchScreeningEvent (e.g. Check the showtimes for Wonder Woman in Paris)
  
2. Slot labeller: slot labeller will be trained seperately for each customer intent to label query tokens with feature categories. (Currently this project only trained slot labeller and support this function for GetWeather intent, this subfunction for the other intents is in development)
  
3. The pulled features will be sent to external API to create response for customer.
 
## Use Case
  
  Input Query: "is it hot in new york city today ?" <br>
  The system will create response: "City:new Weather:light rain Temperate:301.18 Humidity:80"
  
  
  labelling tokens in the query into slot labels that are specifically for the predicted customer intent
  
11784 training data and 600 validation data were used to train Classifer to classify customer query into one of the six customer intents

The project focused on 6 customer intents
* SearchCreativeWork (e.g. Find me the I, Robot television show), 
* GetWeather (e.g. Is it windy in Boston, MA right now?),
* BookRestaurant (e.g. I want to book a highly rated restaurant for me and my boyfriend tomorrow night),
* AddToPlaylist (e.g. Add Diamonds to my roadtrip playlist)
* RateBook (e.g. Give 6 stars to Of Mice and Men)
* SearchScreeningEvent (e.g. Check the showtimes for Wonder Woman in Paris)

When a user input a query, the system would firstly understand the query and classifies it into one of the six intents, then the system would start the slot labelling model that were specifically trained for the predicted intent to pull important features infomation from the query. The pulled feature information would be sent to external API to get a repsonse for the user. (Right now the classifier function is supported for all six intents, slot labelling/getting response function is only supported for GetWeather intent, this function will be supoorted for the other five intents soon)

An example of the system use case:

User input a query: is it hot today in new york city ?
User get a response: City:new york Weather:haze Temperate:290.67 Humidity:53







