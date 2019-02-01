<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\MissingDataImputer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class MissingDataImputerTest extends TestCase
{
    protected $dataset;

    protected $transformer;

    public function setUp()
    {
        $this->dataset = new Unlabeled([
            [30, 'friendly'],
            ['?', 'mean'],
            [50, 'friendly'],
            [60, '?'],
            [10, 'mean'],
        ]);

        $this->transformer = new MissingDataImputer('?');
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(MissingDataImputer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    public function test_fit_transform()
    {
        $this->transformer->fit($this->dataset);

        $this->assertTrue($this->transformer->fitted());

        $this->dataset->apply($this->transformer);

        $this->assertThat($this->dataset[1][0], $this->logicalAnd($this->greaterThan(20), $this->lessThan(55)));
        $this->assertContains($this->dataset[3][1], ['friendly', 'mean']);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->dataset->apply($this->transformer);
    }
}
