const response = await fetch('../map_image.json');
const mapImages = await response.json();

function getMapMatrix(mapName) {
    return mapImages[mapName]; 
}

export function prepareData(teams, brawlers, mapPool, modePool) {
    const { brawlerToId, idToBrawler, idToBrawlerData } = getBrawlers(brawlers);

    const processedData = teams.map(team => {
        const teamIds = [
            brawlerToId[team.brawler1],
            brawlerToId[team.brawler2],
            brawlerToId[team.brawler3]
        ];
        console.log(team)
        const brawlerData = teamIds.map(id => idToBrawlerData[id]);

        const classList = brawlerData.map(b => parseInt(b.classId));
        const hpList = brawlerData.map(b => parseInt(b.hp));
        const atkRangeList = brawlerData.map(b => parseFloat(b.atkRange));
        const speedList = brawlerData.map(b => parseInt(b.speed));

        const totalHP = hpList.reduce((a, b) => a + b, 0);
        const totalRange = atkRangeList.reduce((a, b) => a + b, 0);
        const maxHP = Math.max(...hpList);
        const minHP = Math.min(...hpList);
        const hpSpread = maxHP - minHP;

        const classSet = new Set(classList);
        const allUniqueClasses = classSet.size === 3;
        const hasDuplicateClass = classSet.size < 3;

        const roles = {
            hasTank: hpList.some(hp => hp > 4800),
            hasRange: atkRangeList.some(r => r > 7.0),
            hasSpeedster: speedList.some(s => s > 770),
        };

        const { wrList, prList } = brawlerIdToWR(teamIds, team.map, idToBrawlerData);
        const avgWR = wrList.reduce((a, b) => a + b, 0) / 3;
        const avgPR = prList.reduce((a, b) => a + b, 0) / 3;
        const mapMatrix = getMapMatrix(team.map);

        return {
            teamIds,
            mapMat: mapMatrix,
            wr: parseFloat(team.wr),
            picks: parseInt(team.picks),
            brawlersWr: wrList,
            brawlersPr: prList,
            classIds: classList,
            teamHP: hpList,
            teamAtkRng: atkRangeList,
            teamSpd: speedList,
            totalHP,
            totalRange,
            hpSpread,
            avgWR,
            avgPR,
            allUniqueClasses,
            hasDuplicateClass,
            ...roles
        };
    });

    return { processedData, idToBrawler, idToBrawlerData };
}

export function brawlerIdToWR(teamIds, map, idToBrawlerData) {
    const wrList = [];
    const prList = [];
    const combWr = [];

    teamIds.forEach(brawlerId => {
        const brawlerData = idToBrawlerData[brawlerId];

        if (!brawlerData) {
            console.error(`Missing data for brawler ID: ${brawlerId, brawlerData}`);
            //wrList.push(0.5);
            //prList.push(0.01);
            //combWr.push(0.05); // 0.01 * 0.5 * 10
        }

        // Проверяем корректность ключа карты
        const mapKey = map.toLowerCase().replace(/ /g, '_'); // Нормализация ключа
        const prKey = `${mapKey}_pr`;
        const wrKey = `${mapKey}_wr`;

        // Безопасный парсинг с проверкой NaN
        const rawPr = parseFloat(brawlerData[prKey]);
        const rawWr = parseFloat(brawlerData[wrKey]);

        //  console.log({
        //      mapKey,
        //      prKey,
        //      wrKey,
        //      hasPr: !!brawlerData[prKey],
        //      hasWr: !!brawlerData[wrKey],
        //      rawPr,
        //      rawWr
        //  });

        const pr = Math.max(0.001, rawPr) * 10;
        const wr = Math.min(0.99, Math.max(0.01, rawWr));

        wrList.push(wr);
        prList.push(pr);
    });

    return { wrList, prList, combWr }
}

export function getTensors(data, nuKeys = []) {
    return tf.tidy(() => {
        const tensor1d = (key, type = 'float32') => tf.tensor2d(data.map(d => [d[key]]), [data.length, 1], type);
        const tensorBool = (key) => tf.tensor2d(data.map(d => [d[key] ? 1 : 0]), [data.length, 1], 'int32');

        const floatArray = new Float32Array(data.length * 23 * 35);
        data.forEach((d, i) => {
            const offset = i * 23 * 35;
            for (let y = 0; y < 23; y++) {
                for (let x = 0; x < 35; x++) {
                    floatArray[offset + y * 35 + x] = d.mapMat[y][x] / 255;
                }
            }
        });
        const mapMats = tf.tensor4d(floatArray, [data.length, 23, 35, 1]);

        const brawlersWr = tf.tensor2d(data.map(d => d.brawlersWr), [data.length, 3], 'float32');
        const brawlersPr = tf.tensor2d(data.map(d => d.brawlersPr), [data.length, 3], 'float32');


        const hp = tf.tensor2d(data.map(d => d.teamHP), [data.length, 3], 'int32');
        const speed = tf.tensor2d(data.map(d => d.teamSpd), [data.length, 3], 'int32');
        const atkRange = tf.tensor2d(data.map(d => d.teamAtkRng), [data.length, 3], 'float32');


        const [wr1, wr2, wr3] = tf.split(brawlersWr, 3, 1);
        const [pr1, pr2, pr3] = tf.split(brawlersPr, 3, 1);
        const [hp1, hp2, hp3] = tf.split(hp, 3, 1);
        const [speed1, speed2, speed3] = tf.split(speed, 3, 1);
        const [atkR1, atkR2, atkR3] = tf.split(atkRange, 3, 1);

        function calculateClassCounts(classIds) {
            arr = [0, 0, 0, 0, 0, 0, 0];
            classIds.forEach(id => arr[id] += 1);
            return arr
        }

        const computeStats = (tensor) => ({
            min: tf.min(tensor, 1, true),
            max: tf.max(tensor, 1, true),
            mean: tf.mean(tensor, 1, true)
        });


        const totalStats = tf.tensor2d(data.map);


        const hpStats = computeStats(hp);
        const speedStats = computeStats(speed);
        const atkRangeStats = computeStats(atkRange);

        const wr = tf.tensor1d(data.map(d => d.wr), 'float32');


        const inputs = {
            mapMats,
            wr1, wr2, wr3,
            pr1, pr2, pr3,
            hp1, hp2, hp3,
            speed1, speed2, speed3,
            atkR1, atkR2, atkR3,


            hp_min: hpStats.min,
            hp_max: hpStats.max,
            hp_mean: hpStats.mean,
            speed_min: speedStats.min,
            speed_max: speedStats.max,
            speed_mean: speedStats.mean,
            atkR_min: atkRangeStats.min,
            atkR_max: atkRangeStats.max,
            atkR_mean: atkRangeStats.mean,
            classCounts,
            totalHP: tensor1d('totalHP'),
            totalRange: tensor1d('totalRange'),
            hpSpread: tensor1d('hpSpread'),
            avgWR: tensor1d('avgWR'),
            avgPR: tensor1d('avgPR'),
            allUniqueClasses: tensorBool('allUniqueClasses'),
            hasDuplicateClass: tensorBool('hasDuplicateClass'),
            hasTank: tensorBool('hasTank'),
            hasRange: tensorBool('hasRange'),
            hasSpeedster: tensorBool('hasSpeedster')
        };

        if (nuKeys.length > 0) {
            for (const key of nuKeys) {
                delete inputs[key];
            }
        }

        return {
            inputs: Object.values(inputs),
            outputs: [wr],
        };
    });
}


export function getBrawlers(brawlers) {
    const brawlerToId = {};
    const idToBrawler = {};
    const idToBrawlerData = {};

    brawlers.forEach(brawler => {
        brawlerToId[brawler.name] = parseInt(brawler.id);
        idToBrawler[parseInt(brawler.id)] = brawler.name;
        idToBrawlerData[parseInt(brawler.id)] = brawler;
    });

    return { brawlerToId, idToBrawler, idToBrawlerData };
}