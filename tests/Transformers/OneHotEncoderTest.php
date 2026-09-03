<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Exceptions\RuntimeException;
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

    #[Test]
    public function fitTransform() : void
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

    #[Test]
    public function transformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = [
            ['nice', 'furry', 'friendly'],
        ];

        $this->transformer->transform($samples);
    }

    #[Test]
    public function fitTransformWithExcluded() : void
    {
        $dataset = new Unlabeled([
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ]);

        $this->transformer = new OneHotEncoder(['furry']);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $categories = $this->transformer->categories();

        $this->assertIsArray($categories);
        $this->assertCount(3, $categories);
        $this->assertContainsOnlyArray($categories);

        $dataset->apply($this->transformer);

        $expected = [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $dataset = new Unlabeled(samples: [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $restored = unserialize(serialize($this->transformer));

        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->categories(), $restored->categories());
    }
}
