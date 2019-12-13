<?php

namespace Rubix\ML\Tests\Datasets\Extractors;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Extractors\Extractor;
use Rubix\ML\Datasets\Extractors\CSVWithLabels;
use PHPUnit\Framework\TestCase;

class CSVWithLabelsTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Extractors\CSVWithLabels;
     */
    protected $factory;

    public function setUp() : void
    {
        $this->factory = new CSVWithLabels('tests/labeled.csv', ',', '');
    }

    public function test_build_factory() : void
    {
        $this->assertInstanceOf(CSVWithLabels::class, $this->factory);
        $this->assertInstanceOf(Extractor::class, $this->factory);
    }

    public function test_extract() : void
    {
        $dataset = $this->factory->extract(1);

        $samples = [
            ['nice', 'furry', 'friendly', '4'],
            ['mean', 'furry', 'loner', '-1.5'],
            ['nice', 'rough', 'friendly', '2.6'],
            ['mean', 'rough', 'friendly', '-1'],
            ['nice', 'rough', 'friendly', '2.9'],
            ['nice', 'furry', 'loner', '-5'],
        ];

        $labels = [
            'not monster', 'monster', 'not monster', 'monster',
            'not monster', 'not monster',
        ];

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
        $this->assertEquals($labels, $dataset->labels());
    }
}
