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
 
   The external API used for GetWeather intent: https://openweathermap.org/api
 
## Use Case
  
  Input Query: "is it hot in new york city today ?" <br>
  System response: "City:new york Weather:haze Temperate:290.67 Humidity:53"
 
 How the system created the response according to the input query?
 
  <img width="1102" alt="use case" src="https://github.com/ttwazi/NLP/assets/88044035/83bcea61-a09e-4cfe-aab8-7177a15e86f8">
 
## File in this folder
 
 nlu_preprocessing.ipynb: preprocess the raw dataset into format that's ready to be used to train the models, output data files are saved into cls_data.pickle and weather_seq_data.pickle
  









