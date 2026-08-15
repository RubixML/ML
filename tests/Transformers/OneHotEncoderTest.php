<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\OneHotEncoder;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(OneHotEncoder::class)]
class OneHotEncoderTest extends TestCase
{
    protected OneHotEncoder $transformer;

    protected function setUp() : void
    {
        $this->transformer = new OneHotEncoder();
    }

    public function testFitTransform() : void
    {
        $dataset = new Unlabeled(samples: [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $categories = $this->transformer->categories();

        $this->assertIsArray($categories);
        $this->assertCount(3, $categories);
        $this->assertContainsOnlyArray($categories);

        $dataset->apply($this->transformer);

        $expected = [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }
}
