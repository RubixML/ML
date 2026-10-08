<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\BM25Transformer;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(BM25Transformer::class)]
class BM25TransformerTest extends TestCase
{
    protected BM25Transformer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new BM25Transformer(dampening: 1.2, normalization: 0.75);
    }

    #[Test]
    public function fitTransform() : void
    {
        $dataset = new Unlabeled([
            [1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dfs = $this->transformer->dfs();

        $this->assertIsArray($dfs);
        $this->assertCount(19, $dfs);
        $this->assertContainsOnlyInt($dfs);

        $dataset->apply($this->transformer);

        $expected = [
            [0.9167958406389399, 0.7125097034952160, 0.0, 0.0, 0.4393194544866876, 0.0, 0.0, 0.0, 0.4393194544866876, 0.6166447615704051, 0.0, 0.6166447615704051, 0.0, 0.0, 0.0, 1.6122241206680221, 0.4393194544866876, 0.0, 0.9167958406389399],
            [0.0, 0.5463186515201721, 1.1400876111038365, 0.0, 0.0, 0.7149127716351661, 1.1400876111038365, 0.0, 0.0, 0.0, 0.0, 0.7968858525933337, 0.0, 1.1400876111038365, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9167958406389399, 0.6166447615704051, 0.7125097034952160, 0.0, 0.0, 0.7725617741770452, 0.6166447615704051, 0.0, 0.0, 0.9167958406389399, 0.0, 1.2868479799513850, 0.0, 0.4393194544866876, 0.0, 0.0],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1e-8);
    }

    #[Test]
    public function transformMatchesStandardBm25Formula() : void
    {
        $k1 = 1.2;
        $b = 0.75;

        $samples = [
            [1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0],
        ];

        $transformer = new BM25Transformer(dampening: $k1, normalization: $b);

        $transformer->fit(new Unlabeled($samples));

        $n = count($samples);
        $totalTokens = 0.0;
        $dfs = array_fill(0, count($samples[0]), 0);

        foreach ($samples as $sample) {
            foreach ($sample as $column => $tf) {
                if ($tf > 0) {
                    ++$dfs[$column];

                    $totalTokens += $tf;
                }
            }
        }

        $averageDocumentLength = $totalTokens / $n;

        $expected = [];

        foreach ($samples as $sample) {
            $documentLength = array_sum($sample);

            $row = [];

            foreach ($sample as $column => $tf) {
                if ($tf <= 0) {
                    $row[] = 0.0;

                    continue;
                }

                $idf = log1p(($n - $dfs[$column] + 0.5) / ($dfs[$column] + 0.5));

                $row[] = $idf * ($tf * ($k1 + 1.0) / (
                    $tf + $k1 * (1.0 - $b + $b * $documentLength / $averageDocumentLength)
                ));
            }

            $expected[] = $row;
        }

        $transformer->transform($samples);

        $this->assertEqualsWithDelta($expected, $samples, 1e-8);
    }

    #[Test]
    public function transformSaturatesTermFrequencyAtNumeratorTimesIdf() : void
    {
        $transformer = new BM25Transformer(dampening: 1.2, normalization: 0.0);

        $transformer->fit(new Unlabeled([[1.0], [1.0]]));

        $samples = [[1.0e9]];

        $transformer->transform($samples);

        $idf = log1p((2.0 - 2.0 + 0.5) / (2.0 + 0.5));

        $this->assertEqualsWithDelta([[(1.2 + 1.0) * $idf]], $samples, 1e-8);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = [
            [1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0],
        ];

        $this->transformer->transform($samples);
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $dataset = new Unlabeled([
            [1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $restored = unserialize(serialize($this->transformer));

        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->dfs(), $restored->dfs());
    }
}
