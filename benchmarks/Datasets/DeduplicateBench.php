<?php

namespace Rubix\ML\Benchmarks\Datasets;

use Rubix\ML\Datasets\Labeled;

/**
 * @Groups({"Datasets"})
 * @BeforeMethods({"setUp"})
 */
class DeduplicateBench
{
    protected const SIZES = [
        1000,
        10000,
        100000,
        1000000,
    ];

    /**
     * The datasets indexed by number of samples.
     *
     * @var Labeled[]
     */
    protected array $datasets = [];

    public function setUp() : void
    {
        $base = [];

        for ($i = 0; $i < 250; ++$i) {
            $base[] = [
                'class-' . $i % 3,
                sin($i) * 10.0,
                (float) ($i % 7) / 3.0,
                $i % 25,
            ];
        }

        foreach (self::SIZES as $n) {
            $samples = $labels = [];

            $numRepeats = (int) ceil($n / 250);

            for ($i = 0; $i < $numRepeats; ++$i) {
                foreach ($base as $sample) {
                    $samples[] = $sample;
                    $labels[] = $sample[0];
                }
            }

            $this->datasets[$n] = Labeled::quick($samples, $labels);
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
    public function deduplicate(array $params) : void
    {
        $this->datasets[$params['size']]->deduplicate();
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
