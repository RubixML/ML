<?php

include __DIR__ . '/../vendor/autoload.php';

use Rubix\Engine\CART;
use Rubix\Engine\Prototype;
use Rubix\Engine\Tests\Performance;
use Rubix\Engine\Tests\Accuracy;
use Rubix\Engine\SupervisedDataset;
use Rubix\Engine\Preprocessors\L1Normalizer;
use League\Csv\Reader;

$minSamples = $argv[1] ?? 3;
$maxDepth = $argv[2] ?? 300;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Counterfeit Banknote Detector using CART            ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/banknotes.csv')->setDelimiter(',')->getRecords();

$dataset = new SupervisedDataset(iterator_to_array($dataset));

list ($training, $testing) = $dataset->randomize()->split(0.3);

$pipeline = new Prototype(new CART($minSamples, $maxDepth), [new L1Normalizer()], [new Accuracy(), new Performance()]);

echo 'Training a CART ... ';

$start = microtime(true);

$pipeline->train($training->samples(), $training->outcomes());

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo  "\n";

echo 'Testing model ...' . "\n";

$pipeline->test($testing->samples(), $testing->outcomes());
