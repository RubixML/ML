<?php

namespace Rubix\ML\Benchmarks\Datasets;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;

/**
 * @Groups({"Datasets"})
 * @BeforeMethods({"setUp"})
 */
class SortingBench
{
    protected const SIZES = [
        100,
        1000,
        2500,
        5000,
    ];

    /**
     * The datasets indexed by number of samples.
     *
     * @var \Rubix\ML\Datasets\Labeled[]
     */
    protected $datasets = [];

    public function setUp() : void
    {
        $generator = new Agglomerate([
            'Iris-setosa' => new Blob([5.0, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.1]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]);

        foreach (self::SIZES as $n) {
            $this->datasets[$n] = $generator->generate($n);
        }
    }

    /**
     * @Subject
     * @Iterations(5)
     * @ParamProviders({"provideDatasetSizes"})
     * @OutputTimeUnit("milliseconds", precision=3)
     *
     * @param array{size:int} $params
     */
    public function sort(array $params) : void
    {
        $this->datasets[$params['size']]->sort(function ($recordA, $recordB) {
            return $recordA[1] > $recordB[1];
        });
    }

    /**
     * @return array<string, array{size:int}>
     */
    public function provideDatasetSizes() : array
    {
        $providers = [];

        foreach (self::SIZES as $n) {
            $providers["n={$n}"] = [
                'size' => $n,
            ];
        }

        return $providers;
    }
}
