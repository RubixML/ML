<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\Test;
use Rubix\Engine\Math\Stats;
use Rubix\Engine\Math\Random;
use Rubix\Engine\DecisionForest;
use Rubix\Engine\SupervisedDataset;
use League\Csv\Reader;

$trees = $argv[1] ?? 10;
$ratio = $argv[2] ?? 0.10;
$minSize = $argv[3] ?? 3;
$maxDepth = $argv[4] ?? 300;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Mushroom Classifier using a Decision Forest         ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/mushrooms.csv')->setDelimiter(',');

$dataset = new SupervisedDataset(iterator_to_array($dataset));

list ($training, $testing) = $dataset->randomize()->split(0.5);

$estimator = new DecisionForest($trees, $ratio, $minSize, $maxDepth);

echo 'Training a Decision Forest of ' . $trees . ' trees ... ';

$start = microtime(true);

$estimator->train($training->samples(), $training->outcomes());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

$test = new Test($estimator);

echo 'Testing model ...';

$start = microtime(true);

$accuracy = $test->accuracy($testing->samples(), $testing->outcomes());

echo  "\n";

echo 'Model is ' . (string) $accuracy . '% accurate.' . "\n";

echo  "\n";

echo 'Random Sample Input' . "\n";

$sample = [
    Random::item(['b', 'c', 'x', 'f', 'k', 's']),
    Random::item(['f', 'g', 'y', 's']),
    Random::item(['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']),
    Random::item(['t', 'f']),
    Random::item(['a', 'i', 'c', 'y', 'f', 'm', 'n', 'p', 's']),
    Random::item(['a', 'd', 'f', 'n']),
    Random::item(['c', 'd']),
    Random::item(['b', 'n']),
    Random::item(['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']),
    Random::item(['e', 't']),
    Random::item(['b', 'c', 'u', 'e', 'z', 'r', '?']),
    Random::item(['f', 'y', 'k', 's']),
    Random::item(['f', 'y', 'k', 's']),
    Random::item(['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']),
    Random::item(['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']),
    Random::item(['p', 'u']),
    Random::item(['n', 'o', 'w', 'y']),
    Random::item(['n', 'o', 't']),
    Random::item(['c', 'e', 'f', 'l', 'n', 'p', 's', 'z']),
    Random::item(['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']),
    Random::item(['a', 'c', 'n', 's', 'v', 'y']),
    Random::item(['g', 'l', 'm', 'p', 'u', 'w', 'd']),
];

echo 'Cap Shape: ' . $sample[0] . "\n";
echo 'Cap Surface: ' . $sample[1] . "\n";
echo 'Cap Color: ' . $sample[2] . "\n";
echo 'Bruises: ' . $sample[3] . "\n";
echo 'Odor: ' . $sample[4] . "\n";
echo 'Gill Attachment: ' . $sample[5] . "\n";
echo 'Gill Spacing: ' . $sample[6] . "\n";
echo 'Gill Size: ' . $sample[7] . "\n";
echo 'Gill Color: ' . $sample[8] . "\n";
echo 'Stalk Shape: ' . $sample[9] . "\n";
echo 'Stalk Root: ' . $sample[10] . "\n";
echo 'Stalk Surface Above Ring: ' . $sample[11] . "\n";
echo 'Stalk Surface Below Ring: ' . $sample[12] . "\n";
echo 'Stalk Color Above Ring: ' . $sample[13] . "\n";
echo 'Stalk Color Below Ring: ' . $sample[14] . "\n";
echo 'Veil Type: ' . $sample[15] . "\n";
echo 'Veil Color: ' . $sample[16] . "\n";
echo 'Ring Count: ' . $sample[17] . "\n";
echo 'Ring Type: ' . $sample[18] . "\n";
echo 'Spore Print Color: ' . $sample[19] . "\n";
echo 'Population: ' . $sample[20] . "\n";
echo 'Habitat: ' . $sample[21] . "\n";

echo  "\n";

echo 'Making prediction ... ';

$start = microtime(true);

$prediction = $estimator->predict($sample);

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Outcome: ' . $prediction['outcome'] . "\n";
echo 'Certainty: ' . $prediction['certainty'] . "\n";
