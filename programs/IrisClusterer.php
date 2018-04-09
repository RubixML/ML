<?php

include dirname(__DIR__) . '/vendor/autoload.php';

use Rubix\Engine\DBSCAN;
use Rubix\Engine\Dataset;
use Rubix\Engine\Preprocessors\OneHotEncoder;
use Rubix\Engine\Graph\DistanceFunctions\Euclidean;
use League\Csv\Reader;

$epsilon = $argv[1] ?? 1;
$minDensity = $argv[2] ?? 10;

echo '╔═════════════════════════════════════════════════════╗' . "\n";
echo '║                                                     ║' . "\n";
echo '║ Iris Clusterer using DBSCAN                         ║' . "\n";
echo '║                                                     ║' . "\n";
echo '╚═════════════════════════════════════════════════════╝' . "\n";

$dataset = Reader::createFromPath(dirname(__DIR__) . '/datasets/iris.csv')->setDelimiter(',');

$dataset = Dataset::fromIterator($dataset);

$preprocessor = new OneHotEncoder();

$preprocessor->fit($dataset);
$dataset->transform($preprocessor);

$clusterer = new DBSCAN($epsilon, $minDensity, new Euclidean());

echo 'Clustering samples using DBSCAN ... ';

$start = microtime(true);

$clusters = $clusterer->cluster($dataset);

echo 'done in ' . (string) round(microtime(true) - $start, 5) . ' seconds.' . "\n";

echo "\n";

foreach ($clusters as $label => $samples) {
    echo 'Cluster ' . (string) $label . ': ' . count($samples) . ' has samples' . "\n";
}
