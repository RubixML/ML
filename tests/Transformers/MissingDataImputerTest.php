<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Strategies\Mean;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Strategies\KMostFrequent;
use Rubix\ML\Transformers\MissingDataImputer;
use PHPUnit\Framework\TestCase;

/**
 * @group Transformers
 * @covers \Rubix\ML\Transformers\MissingDataImputer
 */
class MissingDataImputerTest extends TestCase
{
    /**
     * @var MissingDataImputer
     */
    protected $transformer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->transformer = new MissingDataImputer(new Mean(), new KMostFrequent(), '?');
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MissingDataImputer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
    }

    /**
     * @test
     */
    public function fitTransform() : void
    {
        $dataset = new Unlabeled([
            [30, 'friendly'],
            [NAN, 'mean'],
            [50, 'friendly'],
            [60, '?'],
            [10, 'mean'],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dataset->apply($this->transformer);

        $this->assertThat($dataset[1][0], $this->logicalAnd($this->greaterThan(20), $this->lessThan(55)));
        $this->assertContains($dataset[3][1], ['friendly', 'mean']);
    }
}
