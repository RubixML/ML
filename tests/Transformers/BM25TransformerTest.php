<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\BM25Transformer;
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

    public function testFitTransform() : void
    {
        $dataset = new Unlabeled([
            [1, 3, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 4, 1, 0, 1],
            [0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 4, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dfs = $this->transformer->dfs();

        $this->assertIsArray($dfs);
        $this->assertCount(19, $dfs);
        $this->assertContainsOnlyInt($dfs);

        $dataset->apply($this->transformer);

        $expected = [
            [0.2562582002070131, 0.22742881339794754, 0.0, 0.0, 0.13186359514416618, 0.0, 0.0, 0.0, 0.13186359514416618, 0.19254341937443092, 0.0, 0.19254341937443092, 0.0, 0.0, 0.0, 0.4860031535349766, 0.13186359514416618, 0.0, 0.2562582002070131],
            [0.0, 0.17063795450977862, 0.3316106698128093, 0.0, 0.0, 0.23083934808978732, 0.3316106698128093, 0.0, 0.0, 0.0, 0.0, 0.26160416281731713, 0.0, 0.3316106698128093, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2562582002070131, 0.19254341937443092, 0.22742881339794754, 0.0, 0.0, 0.25008418471976107, 0.19254341937443092, 0.0, 0.0, 0.2562582002070131, 0.0, 0.3741808347986538, 0.0, 0.13186359514416618, 0.0, 0.0],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1e-8);
    }
}
