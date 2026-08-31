<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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
            [0.4167253821086091, 0.32386804704328004, 0.0, 0.0, 0.19969066113031256, 0.0, 0.0, 0.0, 0.19969066113031256, 0.2802930734410933, 0.0, 0.2802930734410933, 0.0, 0.0, 0.0, 0.7328291457581919, 0.19969066113031256, 0.0, 0.4167253821086091],
            [0.0, 0.2483266597818964, 0.5182216414108348, 0.0, 0.0, 0.32496035074325735, 0.5182216414108348, 0.0, 0.0, 0.0, 0.0, 0.36222084208787897, 0.0, 0.5182216414108348, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4167253821086091, 0.2802930734410933, 0.32386804704328004, 0.0, 0.0, 0.35116444280774783, 0.2802930734410933, 0.0, 0.0, 0.4167253821086091, 0.0, 0.5849308999779023, 0.0, 0.19969066113031256, 0.0, 0.0],
        ];

        $this->assertEqualsWithDelta($expected, $dataset->samples(), 1e-8);
    }
}
