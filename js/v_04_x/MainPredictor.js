import { getData } from "./dataLoader.js";
import { getTensors, prepareData, brawlerIdToWR, getBrawlers } from "./preparing.js"
import { FilteredPredictor, Predictor } from "./predict.js";

tf.setBackend('webgl').then(() => {
    console.log(tf.getBackend());
});


const Brawlers = 'rankedBrawlers.csv';
const path = '../data/';
const mapPool = {
    'dry_season': 0,
    'hideout': 1,
    'layer_cake': 2,
    'shooting_star': 3,
    'triple_dribble': 4,
    'pinball_dreams': 5,
    'sneaky_fields': 6,
    'center_stage': 7,
    'belles_rock': 8,
    'flaring_phoenix': 9,
    'new_horizons': 10,
    'out_in_the_open': 11,
    'below_zero': 12,
    'cool_box': 13,
    'starr_garden': 14,
    'super_center': 15,
    'parallel_plays': 16,
    'dueling_beetles': 17,
    'open_business': 18,
    'ring_of_fire': 19,
    'double_swoosh': 20,
    'gem_fort': 21,
    'hard_rock_mine': 22,
    'undermine': 23
};
const modePool = {
    'bounty': 0,
    'knockout': 1,
    'brawlball': 2,
    'hockey': 3,
    'hotzone': 4,
    'gemgrab': 5
};

const model = await tf.loadLayersModel('/data/model/BSP-04.03.json');

const brawlerData = await getData(path, Brawlers);
const brawlerNames = brawlerData.map(b => b.name);
const { brawlerToId, idToBrawler, idToBrawlerData } = getBrawlers(brawlerData);

const predictor = new FilteredPredictor({
    maxID: 90,
    idToBrawlerData: idToBrawlerData,
    getTensors: getTensors,
    brawlerIdToWR: brawlerIdToWR,
    mapPool: mapPool,
    modePool: modePool,
    nuKeys: []
});

console.log(brawlerNames)
const bid = brawlerToId

console.log(brawlerData)
export const MainPredictor = {
    predict: function (map, mode, ...team) {
        predictor.reset();
        if (!team || !team.some(a => !a)) {
            predictor.predictMultiple(model,
                [
                    [map, mode],
                ],15
            );
        } else {
            let teamIds = [];
            for(const brawler of team) {
                if(brawler && brawlerNames.indexOf(brawler.toUpperCase()) != -1) {
                    const id = bid[brawler.toUpperCase()];
                    teamIds.push(id);
                }
            }
            predictor.predictWithFilter(
                model,
                teamIds,
                map,
                mode,
                15
            )
        }
    }
}