<?php

namespace Rubix\ML\Tests\Datasets\Extractors;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Extractors\CSV;
use Rubix\ML\Datasets\Extractors\Extractor;
use PHPUnit\Framework\TestCase;

class CSVTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Extractors\CSV;
     */
    protected $factory;

    public function setUp() : void
    {
        $this->factory = new CSV('tests/unlabeled.csv', ',', '');
    }

    public function test_build_factory() : void
    {
        $this->assertInstanceOf(CSV::class, $this->factory);
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

        $this->assertInstanceOf(Unlabeled::class, $dataset);
        $this->assertEquals($samples, $dataset->samples());
    }
}
