# BSPicker v2.0 (ML Edition)

AI-powered Ranked draft assistant for Brawl Stars using Tensorflow.js

## Key features

* **Machine Learning Core**: Predicts team winrate using a custom neural network
* **Real-time Analysis**: Dynamycally updates recomendations based on map/mode optionally brawlers
* **Advanced Statistics**: Considers 30+ features including win rates, pick rates, class balance, and team synergies.

## ML Architecture Highlights

*Model Specifications*

| Feature        | Details                                                              |
|:---------------|:---------------------------------------------------------------------|
| Model Type     | CNN + Dense Network                                                  |
| Input Features | 30+ (Map heatmap, brawler stats, team composition metrics) |
| Output         | Win probability (0-1)                                                |
| Training data  | 50,000 competitive matches (filtered for > 150 picks per team)       |
| Metrics        | R2: 0.62                                                             |

*Technical Stack*

``` mermaid
graph LR
A[CSV Data] --> B[Data Preprocessing]
    B --> C{TensorFlow.js}
    C --> D[CNN for Map Analysis]
    C --> E[Embeddings for Game Modes]
    C --> F[Dense Layers for Team Stats]
    D & E & F --> G[Win Probability Prediction]
```

## Usage Examples

1. Basic Prediction
``` javascript
// predicts top 5 teams for a map
predictor.predictMultiple(model, [['sneaky_fields', 'brawlball']], 5)
```
2. Filtered Draft Analysis
``` javascript
// predicts top 10 teams with Piper
const piperId = brawlerToId["PIPER"];
predictor.predictWithFilter(model, [piperId], "hideout", "bounty");
```
3. Banned Picks Avoidance
``` javascript
// bans annoing brawlers
predictor.predictWithExclusions(model, [mortisId, edgarId, dynamikeId], "ring_of_fire", "hotzone")
```

## Installation & Setup

``` bash
git clone https://github.com/heysonanderson/bsPicker_v2.0.git
cd bsPicker_v2.0
npm install -g http-server
http-server -o
```

## License

MIT License - Includes Tensorflow.js and Papa Parse dependencies