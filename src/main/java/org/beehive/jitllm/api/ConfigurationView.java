package org.beehive.jllm.api;

import org.beehive.jllm.model.Configuration;

/** The internal {@link Configuration}, narrowed to what a user may see. */
final class ConfigurationView implements ModelConfiguration {

    private final Configuration delegate;

    ConfigurationView(Configuration delegate) {
        this.delegate = delegate;
    }

    @Override
    public int dimension() {
        return delegate.dim();
    }

    @Override
    public int hiddenDimension() {
        return delegate.hiddenDim();
    }

    @Override
    public int layers() {
        return delegate.numberOfLayers();
    }

    @Override
    public int attentionHeads() {
        return delegate.numberOfHeads();
    }

    @Override
    public int keyValueHeads() {
        return delegate.numberOfKeyValueHeads();
    }

    @Override
    public int vocabularySize() {
        return delegate.vocabularySize();
    }

    @Override
    public int maxContextLength() {
        return delegate.contextLengthModel();
    }
}
