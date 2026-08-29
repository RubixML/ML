<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Strategies\Mean;
use Rubix\ML\Strategies\KMostFrequent;
use Rubix\ML\Transformers\MissingDataImputer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(MissingDataImputer::class)]
class MissingDataImputerTest extends TestCase
{
    protected MissingDataImputer $transformer;

    protected function setUp() : void
    {
        $this->transformer = new MissingDataImputer(
            continuous: new Mean(),
            categorical: new KMostFrequent(),
            categoricalPlaceholder: '?'
        );
    }

    public function testFitTransform() : void
    {
        $dataset = new Unlabeled(samples: [
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
