const
    globalBrawlers = 'rankedBrawlers.csv',
    mapTeams = 'teamsBigger.csv',
    path = '../data/',
    mapPool = {
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
    },
    modePool = {
        'bounty': 0,
        'knockout': 1,
        'brawlball': 2,
        'hockey': 3,
        'hotzone': 4,
        'gemgrab': 5
    }

import { getData } from "./dataLoader.js";
import { getTensors, prepareData, brawlerIdToWR } from "./preparing.js"

tf.setBackend('webgl').then(() => {
    console.log(tf.getBackend());
});
const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl');
if (!gl) {
    console.error('WebGL не поддерживается!');
} else {
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
        console.log('Вендор:', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
        console.log('Рендерер:', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
    } else {
        console.log('WEBGL_debug_renderer_info не поддерживается');
    }
}


function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

function splitData(data, testSize = 0.2, stratifyBy = 'mapId') {
    // Проверка testSize
    if (testSize < 0 || testSize >= 1) {
        throw new Error('testSize must be between 0 and 1');
    }

    if (data.length === 0) {
        return { train: [], test: [] };
    }
    const grouped = data.reduce((acc, item) => {
        const key = item[stratifyBy];
        if (!acc[key]) acc[key] = [];
        acc[key].push(item);
        return acc;
    }, {});

    const train = [];
    const test = [];
    Object.keys(grouped).forEach(key => {
        const group = grouped[key];
        const splitIdx = Math.floor(group.length * (1 - testSize));

        const shuffledGroup = shuffleArray(group);

        train.push(...shuffledGroup.slice(0, splitIdx));
        test.push(...shuffledGroup.slice(splitIdx));
    });

    return { train, test };
}


export async function main() {
    try {


        //////////////load data

        const [globalStats, teamStats] = await Promise.all([
            getData(path, globalBrawlers),
            getData(path, mapTeams)
        ]);

        //////////filter, prepare, getting tensors

        const filteredTeams = teamStats.filter(team => team.picks >= 150);

        const { processedData, idToBrawler, idToBrawlerData } = prepareData(filteredTeams, globalStats, mapPool, modePool);
        const { train, test } = splitData(processedData, 0.2);

        for (const d of train) {
            if (
                !Array.isArray(d.teamIds) || d.teamIds.some(isNaN) ||
                isNaN(d.mapId) ||
                isNaN(d.modeId) ||
                isNaN(d.wr) ||
                d.brawlersWr.some(isNaN) ||
                d.brawlersPr.some(isNaN) ||
                d.classIds.some(isNaN)
            ) {
                console.error("Ошибка в данных:", d);
                break;
            }
        }

        console.log(processedData.length);

        const keysToDel = []
        const trainData = getTensors(train,keysToDel);
        const testData = getTensors(test,keysToDel);




        ///////////////////create a model


        const model = createModel(
            Object.keys(modePool).length
        );

        //const model = await tf.loadLayersModel('http://127.0.0.1:5500/data/model/BSP-03.01.json');

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanAbsoluteError',
            metrics: ['mae', tf.metrics.r2Score, 'mse']
        })

        console.log(model.summary())
        //  const teamIdsArray = train.map(d => d.mapId);
        //  const teamIdMax = Math.min(...teamIdsArray.flat());
        //  console.log("Максимальный teamId:", teamIdMax);

        const history = await model.fit(
            trainData.inputs,
            trainData.outputs,
            {
                epochs: 300,
                batchSize: 64,
                validationData: [
                    testData.inputs,
                    testData.outputs
                ],
                callbacks: [
                    tf.callbacks.earlyStopping({
                        monitor: 'val_r2Score',
                        patience: 7,
                        mode: 'max',
                        minDelta: 0.001
                    }),
                    new tf.CustomCallback({
                        onEpochEnd: (epoch, logs) => {
                            console.log(`Эпоха ${epoch + 1}: loss = ${logs.loss.toFixed(4)} | mae = ${logs.mae.toFixed(4)} | r2 = ${logs.r2Score ? logs.r2Score.toFixed(4) : 'N/A'} | val_loss = ${logs.val_loss.toFixed(4)} | val_mae = ${logs.val_mae.toFixed(4)} | val_r2 = ${logs.val_r2Score ? logs.val_r2Score.toFixed(4) : 'N/A'}`);
                        }
                    })
                ]
            }
        );


        console.log(history)
        await model.save('downloads://BSP-05.03');
    }
    catch (error) {
        console.error(error);
    }
}

const mul = (...a) => tf.layers.multiply().apply(a)
const dense = (a, units, activation) => tf.layers.dense({ units: units, activation: activation }).apply(a);
const concat = (...b) => tf.layers.concatenate().apply(b);

function createModel(modesCount) {
    const map = tf.input({ shape: [23, 35, 1] });
    const mode = tf.input({ shape: [1] });
    const wr1 = tf.input({ shape: [1] });
    const wr2 = tf.input({ shape: [1] });
    const wr3 = tf.input({ shape: [1] });
    const pr1 = tf.input({ shape: [1] });
    const pr2 = tf.input({ shape: [1] });
    const pr3 = tf.input({ shape: [1] });
    //  const cls1 = tf.input({ shape: [1] });
    //  const cls2 = tf.input({ shape: [1] });
    //  const cls3 = tf.input({ shape: [1] });

    const speed1 = tf.input({ shape: [1] });
    const speed2 = tf.input({ shape: [1] });
    const speed3 = tf.input({ shape: [1] });
    const hp1 = tf.input({ shape: [1] });
    const hp2 = tf.input({ shape: [1] });
    const hp3 = tf.input({ shape: [1] });
    const atkR1 = tf.input({ shape: [1] });
    const atkR2 = tf.input({ shape: [1] });
    const atkR3 = tf.input({ shape: [1] });
    const classCounts = tf.input({ shape: [7] });

    const hp_min = tf.input({shape:[1]});    // [data.length, 1]
    const hp_max = tf.input({shape:[1]});   // [data.length, 1]
    const hp_mean = tf.input({shape:[1]}); // [data.length, 1]
    const speed_min = tf.input({shape:[1]});
    const speed_max = tf.input({shape:[1]});
    const speed_mean = tf.input({shape:[1]});
    const atkR_min = tf.input({shape:[1]});
    const atkR_max = tf.input({shape:[1]});
    const atkR_mean = tf.input({shape:[1]});

    const totalHP = tf.input({shape:[1]});
    const totalRange = tf.input({shape:[1]});
    const hpSpread = tf.input({shape:[1]});
    const avgWR = tf.input({shape:[1]});
    const avgPR = tf.input({shape:[1]});
    const allUniqueClasses = tf.input({shape:[1]});
    const hasDuplicateClass = tf.input({shape:[1]});
    const hasTank = tf.input({shape:[1]});
    const hasRange = tf.input({shape:[1]});
    const hasSpeedster = tf.input({shape:[1]});


    const prepareF = (...f) => {
        return f.map(one => tf.layers.batchNormalization().apply(one))
    }

    const [
        speed1N,
        speed2N,
        speed3N,
        hp1N,
        hp2N,
        hp3N,
        atkR1N,
        atkR2N,
        atkR3N,
        classCountsN,
        hp_minN,
        hp_maxN,
        hp_meanN,
        speed_minN,
        speed_maxN,
        speed_meanN,
        atkR_minN,
        atkR_maxN,
        atkR_meanN,
        totalHPN,
        totalRangeN,
        hpSpreadN,
        avgWRN,
        avgPRN
    ] = prepareF(
        speed1,
        speed2,
        speed3,
        hp1,
        hp2,
        hp3,
        atkR1,
        atkR2,
        atkR3,
        classCounts,
        hp_min,
        hp_max,
        hp_mean,
        speed_min,
        speed_max,
        speed_mean,
        atkR_min,
        atkR_max,
        atkR_mean,
        totalHP,
        totalRange,
        hpSpread,
        avgWR,
        avgPR,       
    )

    // const mapEmbedding = tf.layers.embedding({
    //     inputDim: mapsCount,
    //     outputDim: 12,
    //     name: 'mapEmbedding'
    // }).apply(map);

    const modeEmbedding = tf.layers.embedding({
        inputDim: modesCount,
        outputDim: 8,
        name: 'modeEmbedding'
    }).apply(mode);

    const modeFlatten = tf.layers.flatten().apply(modeEmbedding);

    let mapConv = tf.layers.conv2d({
        filters: 16,
        kernelSize: 5,
        activation: 'relu',
        padding: 'same'
    }).apply(map);
    let mapConv1 = tf.layers.conv2d({
        filters: 16,
        kernelSize: 9,
        activation: 'relu',
        padding: 'same'
    }).apply(map);
    let mapConv2 = tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }).apply(map);

    mapConv = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(mapConv);
    mapConv1 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(mapConv1);
    mapConv2 = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(mapConv2);

    mapConv = tf.layers.maxPooling2d({ poolSize: [2, 2] }).apply(concat(mapConv,mapConv1,mapConv2));

    const mapFlatten = tf.layers.flatten().apply(mapConv);
    const mapCtx = dense(dense(concat(mapFlatten, modeFlatten), 64, 'relu'), 32, 'relu');
    let x = concat(mapCtx,wr1,wr2,wr3,pr1,pr2,pr3, classCountsN, speed1N, speed2N, speed3N, hp1N, hp2N, hp3N, atkR1N, atkR2N, atkR3N,
        hp_minN,
        hp_maxN,
        hp_meanN,
        speed_minN,
        speed_maxN,
        speed_meanN,
        atkR_minN,
        atkR_maxN,
        atkR_meanN,
        totalHPN,
        totalRangeN,
        hpSpreadN,
        avgWRN,
        avgPRN,
        allUniqueClasses,
        hasDuplicateClass,
        hasTank,
        hasRange,
        hasSpeedster
    );

    x = dense(x, 64, 'relu');
    x = dense(x, 32, 'relu');

    const output = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }).apply(x);

    return tf.model({
        inputs: [map, mode,wr1,wr2,wr3,pr1,pr2,pr3,speed1, speed2, speed3, hp1, hp2, hp3, atkR1, atkR2, atkR3, classCounts,
            hp_min,
            hp_max,
            hp_mean,
            speed_min,
            speed_max,
            speed_mean,
            atkR_min,
            atkR_max,
            atkR_mean,
            totalHP,
            totalRange,
            hpSpread,
            avgWR,
            avgPR,
            allUniqueClasses,
            hasDuplicateClass,
            hasTank,
            hasRange,
            hasSpeedster,
        ],
        outputs: output
    });
}

