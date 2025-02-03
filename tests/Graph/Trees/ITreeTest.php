<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Trees;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Graph\Nodes\Depth;
use Rubix\ML\Graph\Trees\ITree;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Trees')]
#[CoversClass(ITree::class)]
class ITreeTest extends TestCase
{
    protected const int DATASET_SIZE = 100;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected ITree $tree;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'east' => new Blob(center: [5, -2, -2]),
                'west' => new Blob(center: [0, 5, -3]),
            ],
            weights: [0.5, 0.5]
        );

        $this->tree = new ITree();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertEquals(0, $this->tree->height());
    }

    public function testGrowSearch() : void
    {
        $this->tree->grow($this->generator->generate(self::DATASET_SIZE));

        $this->assertGreaterThan(5, $this->tree->height());

        $sample = $this->generator->generate(1)->sample(0);

        $node = $this->tree->search($sample);

        $this->assertInstanceOf(Depth::class, $node);
    }

    public function testGrowWithSameSamples() : void
    {
        $generator = new Agglomerate(generators: [
            'east' => new Blob(center: [5, -2, 10], stdDev: 0.0),
        ]);

        $dataset = $generator->generate(self::DATASET_SIZE);

        $this->tree->grow($dataset);

        $this->assertEquals(2, $this->tree->height());
    }
}
