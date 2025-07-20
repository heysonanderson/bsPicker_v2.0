const response = await fetch('../map_image.json');
const mapImages = await response.json();

function getMapMatrix(mapName) {
    return mapImages[mapName];
}

export class Predictor {
    constructor({ maxID, idToBrawlerData, getTensors, brawlerIdToWR, mapPool, modePool, nuKeys }) {
        this.cache = new Map();
        this.combinations = this.generateAllCombinations(maxID)
        this.idToBrawlerData = idToBrawlerData;
        this.getTensors = getTensors;
        this.brawlerIdToWR = brawlerIdToWR;
        this.mapPool = mapPool;
        this.modePool = modePool;
        this.nuKeys = nuKeys;
    }

    generateAllCombinations(maxID) {
        const combinations = [];

        for (let x = 0; x < maxID - 2; x++) {
            for (let y = x + 1; y < maxID - 1; y++) {
                for (let z = y + 1; z < maxID; z++) {
                    combinations.push([x, y, z]);
                }
            }
        }

        return combinations;
    };

    prepareTeamData(teams, mapName, modeName) {

        return teams.map(team => {
            const { wrList, prList } = this.brawlerIdToWR(
                team,
                mapName,
                this.idToBrawlerData
            );

            const syntheticWR = wrList.reduce((sum, wr) => sum + wr, 0) / 3;
            const classList = team.map(id => parseInt(this.idToBrawlerData[id].classId));
            const hpList = team.map(id => parseInt(this.idToBrawlerData[id].hp));
            const atkRangeList = team.map(id => parseFloat(this.idToBrawlerData[id].atkRange));
            const speedList = team.map(id => parseInt(this.idToBrawlerData[id].speed));

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

            const avgWR = wrList.reduce((a, b) => a + b, 0) / 3;
            const avgPR = prList.reduce((a, b) => a + b, 0) / 3;
            const mapMatrix = getMapMatrix(mapName);

            return {
                teamIds: team,
                mapMat: mapMatrix,
                modeId: this.modePool[modeName],
                wr: syntheticWR,
                picks: 1,
                brawlersWr: wrList,
                brawlersPr: prList,
                classIds: classList,
                teamHP: hpList,
                teamAtkRng: atkRangeList,
                teamSpd: speedList,
                classIds: classList,
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
    };

    predictMultiple(model, maps, first) {
        if (!model) throw new Error("Model not provided");
        return Promise.all(
            maps.map(([map, mode]) => this.predictForMap(model, map, mode, first))
        );
    };

    async cachedPredict(model, mapName, modeName, first) {
        const key = `${mapName}|${modeName}`;
        if (!this.cache.has(key)) {
            const result = await this.predictForMap(model, mapName, modeName, first);
            this.cache.set(key, result);
        }
        return this.cache.get(key);
    };

    async predictForMap(model, mapName, modeName, firstOf = 10) {
        const preparedData = this.prepareTeamData(
            this.combinations,
            mapName,
            modeName
        );

        const inputs = this.getTensors(preparedData, this.nuKeys).inputs;
        const predictions = await model.predict(inputs).data();

        return this.proPredict(preparedData, predictions, mapName, firstOf);
    }

    proPredict(data, predictions, mapName, firstOf = 10) {
        const results = data.map((team, index) => ({
            brawlers: team.teamIds,
            names: this.idToName(team.teamIds),
            predictedWR: predictions[index]
        }));

        results.sort((a, b) => b.predictedWR - a.predictedWR);

        const recommends = {};
        for (let i = 0; i < results.length; i++) {
            const team = results[i];
            team.names.forEach(name => {
                if (!recommends[name]) {
                    recommends[name] = 0; // Инициализируем нулем, если персонаж встречается первый раз
                }
                recommends[name] += team.predictedWR;
            });
        }

        // Преобразуем объект в массив и сортируем по убыванию винрейта
        const sortedRecommendations = Object.entries(recommends).slice(0, 15)
            .map(([name, wr]) => ({ name, wr }))
            .sort((a, b) => b.wr - a.wr);

        // Вывод рекомендаций
        console.log(`Рекомендуемые персонажи на карте ${mapName}(по сумме винрейтов):`);
        sortedRecommendations.forEach((item, index) => {
            console.log(`${index + 1}. ${item.name}: ${item.wr.toFixed(2)}%`);
        });


        console.log(`Топ-${firstOf} команд на карте ${mapName}`);
        results.slice(0, firstOf).forEach((team, i) => {
            console.log(`${i + 1}. [${team.names.join(', ')}]: ${team.predictedWR.toFixed(4)}`);
        });

        return results.slice(0, firstOf);
    }

    idToName(teamIds) {
        return teamIds.map(id => this.idToBrawlerData[id].name || `Unknown_${id}`);
    }
}

export class FilteredPredictor extends Predictor {
    async predictWithFilter(model, selectedBrawlerIds, mapName, modeName, firstOf = 10) {
        // 1. Фильтруем комбинации, содержащие выбранного бравлера
        for (const selectedBrawlerId of selectedBrawlerIds) {
            this.combinations = this.combinations.filter(team =>
                team.includes(selectedBrawlerId)
            );
        };

        // 2. Подготавливаем данные только для этих комбинаций
        const preparedData = this.prepareTeamData(
            this.combinations,
            mapName,
            modeName
        );

        // 3. Получаем предсказания
        const inputs = this.getTensors(preparedData, this.nuKeys).inputs;
        const predictions = await model.predict(inputs).data();

        this.proPredict(preparedData, predictions, mapName, firstOf);
    }

    // Альтернативная версия с исключением нежелательных бравлеров
    async predictWithExclusions(model, excludeIds, mapName, modeName, firstOf = 10) {
        this.combinations = this.combinations.filter(team =>
            !team.some(id => excludeIds.includes(id))
        );

        const preparedData = this.prepareTeamData(
            this.combinations,
            mapName,
            modeName
        );

        // 3. Получаем предсказания
        const inputs = this.getTensors(preparedData, this.nuKeys).inputs;
        const predictions = await model.predict(inputs).data();

        this.proPredict(preparedData, predictions, mapName, firstOf);

    }
}