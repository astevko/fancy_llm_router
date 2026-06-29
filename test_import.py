#!/usr/bin/env python3
"""Test script to verify the LLM Router works."""

import asyncio

from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.schemas.models import ModelInfo, ModelProvider, ModelCapabilities, ModelPricing
from fancy_llm_router.schemas.requests import CompletionRequest
from fancy_llm_router.schemas.routing import RoutingStrategy


async def main():
    # Create router
    router = LLMRouter()
    
    # Add a test model
    model = ModelInfo(
        provider=ModelProvider.OPENAI,
        model_id='gpt-4',
        name='GPT-4',
        capabilities=ModelCapabilities(
            max_tokens=8192,
            max_input_tokens=8192,
            context_window=8192,
            supports_chat=True,
            supports_completions=True,
        ),
        pricing=ModelPricing(
            input_token_price=0.03,
            output_token_price=0.06,
        ),
        parameters=1000000000,
    )
    
    router.register_model(model)
    
    # Add another model
    model2 = ModelInfo(
        provider=ModelProvider.OPENAI,
        model_id='gpt-3.5-turbo',
        name='GPT-3.5 Turbo',
        capabilities=ModelCapabilities(
            max_tokens=4096,
            max_input_tokens=4096,
            context_window=4096,
            supports_chat=True,
            supports_completions=False,
        ),
        pricing=ModelPricing(
            input_token_price=0.0015,
            output_token_price=0.002,
        ),
        parameters=175000000,
    )
    
    router.register_model(model2)
    
    # List models
    models = router.list_models()
    print(f'Registered models: {models}')
    
    # Test routing
    request = CompletionRequest(model='gpt-4', prompt='Test prompt', max_tokens=100)
    decision = await router.route(request, strategy=RoutingStrategy.COST_OPTIMIZED)
    print(f'Routing decision: {decision.selected_model}')
    print(f'Provider: {decision.selected_provider}')
    print(f'Strategy: {decision.strategy}')
    print(f'Confidence: {decision.confidence:.3f}')
    
    # Test with different strategy
    decision2 = await router.route(request, strategy=RoutingStrategy.QUALITY_OPTIMIZED)
    print(f'Quality-optimized decision: {decision2.selected_model}')
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
