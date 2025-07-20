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
    };

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

    // Обработка пустых данных
    if (data.length === 0) {
        return { train: [], test: [] };
    }

    // Группируем данные по значению признака (stratifyBy)
    const grouped = data.reduce((acc, item) => {
        const key = item[stratifyBy]; // Значение признака для группировки
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
const dropout = (dense) => tf.layers.dropout({ rate: 0.3 }).apply(dense);
const max = (...b) => tf.layers.maximum().apply(b);
const flatt = (a) => tf.layers.flatten().apply(a);
const attention = (brawler1, brawler2) => {
    const units = brawler1.shape[1];

    // 1. Совместное представление
    const combined = tf.layers.concatenate().apply([brawler1, brawler2]);

    // 2. Вычисление весов
    const weights = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: 'zeros' // Инициализация для стабильности
    }).apply(combined);

    // 3. Взвешенные признаки
    const weighted1 = tf.layers.multiply().apply([brawler1, weights]);
    const weighted2 = tf.layers.multiply().apply([brawler2, weights]);

    return tf.layers.concatenate().apply([weighted1, weighted2]);
}
function multiHeadAttention(inputs, headSize, numHeads) {
    const embeddingDim = inputs.shape[2];
    if (embeddingDim !== headSize * numHeads) {
        throw new Error(`embeddingDim (${embeddingDim}) должен быть равен headSize * numHeads (${headSize * numHeads})`);
    }

    const outputs = [];
    for (let i = 0; i < numHeads; i++) {
        // Применяем selfAttention для каждой головы
        const headOutput = selfAttention(inputs, headSize); // [batch, seqLength, headSize]
        outputs.push(headOutput);
    }

    // Конкатенация результатов всех голов
    const concatenated = tf.layers.concatenate({ axis: -1 }).apply(outputs); // [batch, seqLength, headSize * numHeads]

    // Финальная проекция для восстановления исходной размерности
    return tf.layers.dense({ units: embeddingDim }).apply(concatenated); // [batch, seqLength, embeddingDim]
}
function selfAttention(inputs, headSize) {
    const seqLength = inputs.shape[1];

    const query = tf.layers.dense({ units: headSize }).apply(inputs); // [batch, seqLength, headSize]
    const key = tf.layers.dense({ units: headSize }).apply(inputs);   // [batch, seqLength, headSize]
    const value = tf.layers.dense({ units: headSize }).apply(inputs); // [batch, seqLength, headSize]

    // Приближаем scores через поэлементное умножение и dense слой
    const scores = tf.layers.dense({
        units: seqLength,
        activation: 'linear',
        weights: [
            tf.randomNormal([headSize, seqLength]), // Имитация Q × Kᵀ
            tf.zeros([seqLength])
        ]
    }).apply(tf.layers.multiply().apply([query, key])); // [batch, seqLength, seqLength]

    // Нормализация: умножаем на фиксированный коэффициент 1/sqrt(headSize)
    const scaleFactor = 1 / Math.sqrt(headSize); // Вычисляем на стороне JS
    const scaledScores = tf.layers.multiply().apply([
        scores,
        tf.layers.dense({
            units: seqLength,
            activation: 'linear',
            weights: [tf.fill([seqLength, seqLength], scaleFactor), tf.zeros([seqLength])]
        }).apply(scores)
    ]);

    // Применяем softmax
    const weights = tf.layers.softmax({ axis: -1 }).apply(scaledScores); // [batch, seqLength, seqLength]

    // Применяем weights к value через dense слой
    const output = tf.layers.dense({
        units: headSize,
        activation: 'linear',
        weights: [
            tf.randomNormal([seqLength, headSize]), // Имитация Weights × V
            tf.zeros([headSize])
        ]
    }).apply(weights); // [batch, seqLength, headSize]

    return output;
}
const prepareF = (...f) => {
    return f.map(one => tf.layers.batchNormalization().apply(one))
}
function createModel(modesCount) {
    

}